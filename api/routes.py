import asyncio
import re
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.mysql import (
    Candidate,
    Job,
    get_db,
    get_or_create_company,
    prefilter_candidate_ids_for_job,
    prefilter_job_ids_for_candidate,
)
from database.neo4j import graph_client
from database.vector_db import vector_client
from ingestion.parser import chunk_text
from ingestion.skill_extractor import skill_extractor
from services.cache import cache_client
from services.embedding import embedding_service
from services.graph_search import graph_search_service
from services.hybrid_scoring import hybrid_scorer
from services.lexical_search import lexical_search_service
from services.rrf import rrf_fuse
from services.reranker import reranker
from services.vector_search import candidate_to_text, job_to_text, vector_search_service

settings = get_settings()
router = APIRouter()


class IngestCandidateRequest(BaseModel):
    name: str
    location: str | None = None
    years_experience: float | None = None
    salary_expectation: int | None = None
    degree: str | None = None
    job_status: str = "open_to_work"
    resume_summary: str = ""
    project_experience: str = ""
    achievements: str = ""
    skills: list[str] | None = None
    company_name: str | None = None
    company_industry: str | None = None


class IngestJobRequest(BaseModel):
    title: str
    company_id: int | None = None
    company_name: str | None = None
    company_industry: str | None = None
    location: str | None = None
    salary_range: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    degree_required: str | None = None
    status: str = "open"
    job_description: str = ""
    responsibilities: str = ""
    preferred_qualifications: str = ""
    skills: list[str] | None = None


class SearchCandidatesRequest(BaseModel):
    job_id: int = Field(..., gt=0)


class SearchJobsRequest(BaseModel):
    candidate_id: int = Field(..., gt=0)


class CandidateSearchResult(BaseModel):
    candidate_id: int
    score: float
    reason: str


class JobSearchResult(BaseModel):
    job_id: int
    score: float
    reason: str


@router.get("/candidates/{candidate_id}/name")
def get_candidate_name(candidate_id: int, db: Session = Depends(get_db)) -> dict:
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="candidate not found")
    return {"candidate_id": candidate.id, "name": candidate.name}


def _parse_salary_range(salary_range: str | None) -> tuple[int | None, int | None]:
    if not salary_range:
        return None, None
    numbers = [int(num) for num in re.findall(r"\d+", salary_range)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def _candidate_score_map(items: list[dict]) -> Dict[int, float]:
    return {int(item["candidate_id"]): float(item.get("score", 0.0)) for item in items}


def _job_score_map(items: list[dict]) -> Dict[int, float]:
    return {int(item["job_id"]): float(item.get("score", 0.0)) for item in items}


def _build_candidate_contexts(candidates: list[Candidate]) -> list:
    return [hybrid_scorer.build_candidate_context(candidate, allow_llm=False) for candidate in candidates]


def _build_job_contexts(jobs: list[Job]) -> list:
    return [hybrid_scorer.build_job_context_light(job, allow_llm=False) for job in jobs]


def _dedupe_candidate_contexts_by_name(candidate_contexts: list) -> list:
    chosen: dict[str, object] = {}

    for ctx in candidate_contexts:
        key = re.sub(r"\s+", "", str(getattr(ctx.candidate, "name", "") or "").strip())
        if not key:
            key = f"candidate:{ctx.candidate.id}"

        current = chosen.get(key)
        if current is None:
            chosen[key] = ctx
            continue

        current_score = ((getattr(current, "years", 0.0) or 0.0), len(getattr(current, "text", "") or ""), current.candidate.id)
        new_score = ((getattr(ctx, "years", 0.0) or 0.0), len(getattr(ctx, "text", "") or ""), ctx.candidate.id)
        if new_score > current_score:
            chosen[key] = ctx

    return list(chosen.values())


def _dedupe_job_contexts(job_contexts: list) -> list:
    chosen: dict[str, object] = {}

    for ctx in job_contexts:
        title = str(getattr(ctx.job, "title", "") or "").strip().lower()
        core_text = (str(getattr(ctx.job, "job_description", "") or "") + str(getattr(ctx.job, "responsibilities", "") or ""))[:200]
        key = re.sub(r"\s+", "", f"{title}|{core_text}")
        if not key:
            key = f"job:{ctx.job.id}"

        current = chosen.get(key)
        if current is None or ctx.job.id > current.job.id:
            chosen[key] = ctx

    return list(chosen.values())


@router.post("/ingest/candidate")
def ingest_candidate(payload: IngestCandidateRequest, db: Session = Depends(get_db)) -> dict:
    company = None
    try:
        if payload.company_name:
            company = get_or_create_company(db, payload.company_name, payload.company_industry)

        candidate = Candidate(
            name=payload.name,
            location=payload.location,
            years_experience=payload.years_experience,
            salary_expectation=payload.salary_expectation,
            degree=payload.degree,
            job_status=payload.job_status,
            resume_summary=payload.resume_summary,
            project_experience=payload.project_experience,
            achievements=payload.achievements,
        )
        db.add(candidate)
        db.flush()

        candidate_text = candidate_to_text(candidate)
        skills = payload.skills or skill_extractor.extract_skills(candidate_text)
        chunks = chunk_text(candidate_text)
        vectors = embedding_service.embed_texts(chunks)

        graph_client.upsert_candidate_skills(candidate.id, skills)
        if company:
            graph_client.link_candidate_company(candidate.id, company.id, company.name)
        vector_client.upsert_candidate_chunks(candidate.id, vectors, chunks)

        db.commit()
        return {
            "candidate_id": candidate.id,
            "skills": skills,
            "chunk_count": len(chunks),
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"candidate ingestion failed: {exc}") from exc


@router.post("/ingest/job")
def ingest_job(payload: IngestJobRequest, db: Session = Depends(get_db)) -> dict:
    company = None
    try:
        salary_min = payload.salary_min
        salary_max = payload.salary_max
        if salary_min is None and salary_max is None and payload.salary_range:
            salary_min, salary_max = _parse_salary_range(payload.salary_range)

        company_id = payload.company_id
        if payload.company_name:
            company = get_or_create_company(db, payload.company_name, payload.company_industry)
            company_id = company.id

        job = Job(
            title=payload.title,
            company_id=company_id,
            location=payload.location,
            salary_range=payload.salary_range,
            salary_min=salary_min,
            salary_max=salary_max,
            degree_required=payload.degree_required,
            status=payload.status,
            job_description=payload.job_description,
            responsibilities=payload.responsibilities,
            preferred_qualifications=payload.preferred_qualifications,
        )
        db.add(job)
        db.flush()

        full_job_text = job_to_text(job)
        skills = payload.skills or skill_extractor.extract_skills(full_job_text)
        chunks = chunk_text(full_job_text)
        vectors = embedding_service.embed_texts(chunks)

        graph_client.upsert_job_skills(job.id, skills)
        if company_id:
            company_name = payload.company_name or "unknown-company"
            graph_client.link_job_company(job.id, company_id, company_name)
        vector_client.upsert_job_chunks(job.id, vectors, chunks)

        db.commit()
        return {
            "job_id": job.id,
            "skills": skills,
            "chunk_count": len(chunks),
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"job ingestion failed: {exc}") from exc


@router.post("/search/candidates", response_model=List[CandidateSearchResult])
async def search_candidates(payload: SearchCandidatesRequest, db: Session = Depends(get_db)) -> List[CandidateSearchResult]:
    cache_key = f"search:candidates:{payload.job_id}"
    cached = cache_client.get_json(cache_key)
    if cached:
        return cached

    job = db.get(Job, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    job_ctx = await asyncio.to_thread(hybrid_scorer.build_job_context, job, True)

    candidate_filter_ids = prefilter_candidate_ids_for_job(db, job, settings.prefilter_limit)

    if candidate_filter_ids:
        candidate_pool = db.scalars(select(Candidate).where(Candidate.id.in_(candidate_filter_ids))).all()
    else:
        candidate_pool = db.scalars(select(Candidate).limit(settings.prefilter_limit)).all()

    if not candidate_pool:
        return []

    candidate_contexts = await asyncio.to_thread(_build_candidate_contexts, candidate_pool)
    filtered_contexts = hybrid_scorer.hard_filter_candidate_contexts(job_ctx, candidate_contexts)
    filtered_contexts = _dedupe_candidate_contexts_by_name(filtered_contexts)
    if not filtered_contexts:
        return []

    filtered_candidate_ids = [ctx.candidate.id for ctx in filtered_contexts]
    lexical_candidates = [ctx.candidate for ctx in filtered_contexts]

    graph_task = asyncio.to_thread(
        graph_search_service.retrieve_candidates_for_job,
        job.id,
        filtered_candidate_ids,
    )
    vector_task = asyncio.to_thread(
        vector_search_service.retrieve_candidates_for_job,
        job,
        filtered_candidate_ids,
    )
    lexical_task = asyncio.to_thread(
        lexical_search_service.retrieve_candidates_for_job,
        job,
        lexical_candidates,
        settings.graph_top_k,
    )

    graph_result, vector_result, lexical_result = await asyncio.gather(graph_task, vector_task, lexical_task)

    graph_rank = [item["candidate_id"] for item in graph_result]
    vector_rank = [item["candidate_id"] for item in vector_result]
    lexical_rank = [item["candidate_id"] for item in lexical_result]

    shortlist_top_n = max(settings.llm_rerank_top_n * 4, settings.rerank_top_n, len(filtered_candidate_ids))
    fused = rrf_fuse(
        {"graph": graph_rank, "vector": vector_rank, "lexical": lexical_rank},
        k=settings.rrf_k,
        top_n=shortlist_top_n,
    )

    if fused:
        candidate_ids = [candidate_id for candidate_id, _ in fused]
    else:
        candidate_ids = filtered_candidate_ids[:shortlist_top_n]

    context_map = {ctx.candidate.id: ctx for ctx in filtered_contexts}
    vector_score_map = _candidate_score_map(vector_result)
    graph_score_map = _candidate_score_map(graph_result)

    pairs: list[dict] = []
    for candidate_id in candidate_ids:
        candidate_ctx = context_map.get(candidate_id)
        if not candidate_ctx:
            continue
        signal = hybrid_scorer.score_candidate_for_job(
            job_ctx,
            candidate_ctx,
            vector_raw=vector_score_map.get(candidate_id),
            graph_raw=graph_score_map.get(candidate_id),
        )
        pairs.append(
            {
                "candidate_id": candidate_id,
                "candidate_text": candidate_ctx.text,
                "hybrid_score": signal.final_score,
                "hybrid_reason": signal.reason,
                "signal_prompt": signal.prompt_context,
            }
        )

    if not pairs:
        return []

    pairs = sorted(pairs, key=lambda item: item["hybrid_score"], reverse=True)
    llm_pairs = pairs[: min(settings.llm_rerank_top_n, len(pairs))]

    reranked = await asyncio.to_thread(reranker.rerank_candidates_for_job, job_ctx.text, llm_pairs)
    final_results = reranked[: min(settings.rerank_top_n, len(reranked))]

    cache_client.set_json(cache_key, final_results, ttl_seconds=120)
    return final_results


@router.post("/search/jobs", response_model=List[JobSearchResult])
async def search_jobs(payload: SearchJobsRequest, db: Session = Depends(get_db)) -> List[JobSearchResult]:
    cache_key = f"search:jobs:{payload.candidate_id}"
    cached = cache_client.get_json(cache_key)
    if cached:
        return cached

    candidate = db.get(Candidate, payload.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="candidate not found")

    candidate_ctx = await asyncio.to_thread(hybrid_scorer.build_candidate_context, candidate, True)

    job_filter_ids = prefilter_job_ids_for_candidate(db, candidate, settings.prefilter_limit)

    if job_filter_ids:
        job_pool = db.scalars(select(Job).where(Job.id.in_(job_filter_ids))).all()
    else:
        job_pool = db.scalars(select(Job).limit(settings.prefilter_limit)).all()

    if not job_pool:
        return []

    job_contexts = await asyncio.to_thread(_build_job_contexts, job_pool)
    filtered_job_contexts = hybrid_scorer.hard_filter_job_contexts(candidate_ctx, job_contexts)
    filtered_job_contexts = _dedupe_job_contexts(filtered_job_contexts)
    if not filtered_job_contexts:
        return []

    filtered_job_ids = [ctx.job.id for ctx in filtered_job_contexts]
    lexical_jobs = [ctx.job for ctx in filtered_job_contexts]

    graph_task = asyncio.to_thread(
        graph_search_service.retrieve_jobs_for_candidate,
        candidate.id,
        filtered_job_ids,
    )
    vector_task = asyncio.to_thread(
        vector_search_service.retrieve_jobs_for_candidate,
        candidate,
        filtered_job_ids,
    )
    lexical_task = asyncio.to_thread(
        lexical_search_service.retrieve_jobs_for_candidate,
        candidate,
        lexical_jobs,
        settings.graph_top_k,
    )

    graph_result, vector_result, lexical_result = await asyncio.gather(graph_task, vector_task, lexical_task)

    graph_rank = [item["job_id"] for item in graph_result]
    vector_rank = [item["job_id"] for item in vector_result]
    lexical_rank = [item["job_id"] for item in lexical_result]

    shortlist_top_n = max(settings.llm_rerank_top_n * 4, settings.rerank_top_n, len(filtered_job_ids))
    fused = rrf_fuse(
        {"graph": graph_rank, "vector": vector_rank, "lexical": lexical_rank},
        k=settings.rrf_k,
        top_n=shortlist_top_n,
    )

    if fused:
        job_ids = [job_id for job_id, _ in fused]
    else:
        job_ids = filtered_job_ids[:shortlist_top_n]

    context_map = {ctx.job.id: ctx for ctx in filtered_job_contexts}
    vector_score_map = _job_score_map(vector_result)
    graph_score_map = _job_score_map(graph_result)

    pairs: list[dict] = []
    for job_id in job_ids:
        job_ctx = context_map.get(job_id)
        if not job_ctx:
            continue
        signal = hybrid_scorer.score_job_for_candidate(
            candidate_ctx,
            job_ctx,
            vector_raw=vector_score_map.get(job_id),
            graph_raw=graph_score_map.get(job_id),
        )
        pairs.append(
            {
                "job_id": job_id,
                "job_text": job_ctx.text,
                "hybrid_score": signal.final_score,
                "hybrid_reason": signal.reason,
                "signal_prompt": signal.prompt_context,
            }
        )

    if not pairs:
        return []

    pairs = sorted(pairs, key=lambda item: item["hybrid_score"], reverse=True)
    llm_pairs = pairs[: min(settings.llm_rerank_top_n, len(pairs))]

    reranked = await asyncio.to_thread(reranker.rerank_jobs_for_candidate, candidate_ctx.text, llm_pairs)
    final_results = reranked[: min(settings.rerank_top_n, len(reranked))]

    cache_client.set_json(cache_key, final_results, ttl_seconds=120)
    return final_results
