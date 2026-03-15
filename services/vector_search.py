from database.mysql import Candidate, Job
from database.vector_db import vector_client
from services.embedding import embedding_service
from config.settings import get_settings

settings = get_settings()


def candidate_to_text(candidate: Candidate) -> str:
    return "\n".join(
        part
        for part in [candidate.resume_summary, candidate.project_experience, candidate.achievements]
        if part
    )


def job_to_text(job: Job) -> str:
    return "\n".join(
        part
        for part in [job.job_description, job.responsibilities, job.preferred_qualifications]
        if part
    )


class VectorSearchService:
    def retrieve_candidates_for_job(self, job: Job, candidate_ids: list[int] | None = None) -> list[dict]:
        query_text = job_to_text(job)
        if not query_text:
            return []
        query_vector = embedding_service.embed_text(query_text)
        return vector_client.search_candidates(
            query_vector=query_vector,
            top_k=settings.vector_top_k,
            candidate_ids=candidate_ids,
        )

    def retrieve_jobs_for_candidate(self, candidate: Candidate, job_ids: list[int] | None = None) -> list[dict]:
        query_text = candidate_to_text(candidate)
        if not query_text:
            return []
        query_vector = embedding_service.embed_text(query_text)
        return vector_client.search_jobs(
            query_vector=query_vector,
            top_k=settings.vector_top_k,
            job_ids=job_ids,
        )


vector_search_service = VectorSearchService()
