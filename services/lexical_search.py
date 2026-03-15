import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from config.settings import get_settings
from database.mysql import Candidate, Job

settings = get_settings()

_SEED_DISCIPLINE_PROFILES = {
    "structure": ["结构", "结构设计", "结构开发", "xpm", "主设", "整机", "机壳", "模具", "dfm", "公差", "注塑", "压铸"],
    "mbb": ["mbb", "cpe", "网关", "驱动", "网卡", "phy", "gmac", "pcie", "pci", "linux", "tcpdump", "gdb"],
    "rf": ["射频", "rf", "天线", "4g", "5g", "高通", "mtk", "展锐", "3gpp", "lte", "频谱仪", "示波器", "综测仪"],
    "backend": ["后端", "python", "java", "golang", "redis", "mysql", "postgresql", "fastapi", "django", "spring"],
    "data": ["数据分析", "机器学习", "深度学习", "pytorch", "tensorflow", "xgboost", "spark", "hadoop"],
    "frontend": ["前端", "react", "vue", "typescript", "javascript", "webpack", "css", "html"],
    "product": ["产品经理", "需求分析", "roadmap", "用户调研", "prd", "竞品分析"],
}

_STOPWORDS = {
    "负责",
    "熟悉",
    "能够",
    "具有",
    "相关",
    "经验",
    "优先",
    "岗位",
    "工作",
    "项目",
    "参与",
    "完成",
    "良好",
    "团队",
    "以上",
    "以下",
    "要求",
    "能力",
}

_MAX_DYNAMIC_TERMS = 80
_MAX_DYNAMIC_PROFILE_TERMS = 32


def _normalize_term(term: str) -> str:
    return str(term or "").strip().lower()


def _normalize_terms(terms: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        token = _normalize_term(term)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _resolve_taxonomy_path() -> Path | None:
    path_text = str(getattr(settings, "domain_taxonomy_file", "") or "").strip()
    if not path_text:
        return None

    path = Path(path_text)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parent.parent
    return (project_root / path).resolve()


def _parse_external_profiles(raw: object) -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = {}

    if isinstance(raw, dict):
        if isinstance(raw.get("profiles"), dict):
            raw = raw["profiles"]

        if isinstance(raw, dict):
            for name, terms in raw.items():
                if isinstance(terms, list):
                    normalized = _normalize_terms([str(item) for item in terms])
                else:
                    normalized = _normalize_terms([str(terms)])
                if normalized:
                    profiles[_normalize_term(str(name))] = normalized
            return profiles

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = _normalize_term(str(item.get("name", "")))
            terms = item.get("terms", [])
            if not name:
                continue
            if isinstance(terms, list):
                normalized = _normalize_terms([str(token) for token in terms])
            else:
                normalized = _normalize_terms([str(terms)])
            if normalized:
                profiles[name] = normalized

    return profiles


@lru_cache(maxsize=1)
def _load_external_profiles() -> dict[str, list[str]]:
    path = _resolve_taxonomy_path()
    if path is None or not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return _parse_external_profiles(raw)


def _all_profiles() -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}

    for source in (_SEED_DISCIPLINE_PROFILES, _load_external_profiles()):
        for name, terms in source.items():
            key = _normalize_term(name)
            if not key:
                continue
            existing = merged.get(key, [])
            merged[key] = _normalize_terms(existing + [str(term) for term in terms])

    return merged


@lru_cache(maxsize=1)
def _all_profile_terms() -> tuple[str, ...]:
    terms: list[str] = []
    for profile_terms in _all_profiles().values():
        terms.extend(profile_terms)
    return tuple(_normalize_terms(terms))


def _candidate_text(candidate: Candidate) -> str:
    return "\n".join(
        part
        for part in [candidate.resume_summary, candidate.project_experience, candidate.achievements]
        if part
    )


def _job_text(job: Job) -> str:
    return "\n".join(
        part
        for part in [job.job_description, job.responsibilities, job.preferred_qualifications]
        if part
    )


def _extract_chinese_segments(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}", text)


def _extract_chinese_ngrams(segments: list[str], min_n: int = 2, max_n: int = 4) -> set[str]:
    grams: set[str] = set()
    for seg in segments:
        if len(seg) > 30:
            continue
        length = len(seg)
        for n in range(min_n, max_n + 1):
            if length < n:
                continue
            for i in range(length - n + 1):
                grams.add(seg[i : i + n])
    return grams


def _keyword_counter(text: str) -> Counter[str]:
    lower = text.lower()
    counter: Counter[str] = Counter()

    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\+\.#-]+", lower):
        if len(token) < 2:
            continue
        counter[token] += 1

    segments = _extract_chinese_segments(text)
    for seg in segments:
        if seg in _STOPWORDS:
            continue
        counter[seg] += 1

    for gram in _extract_chinese_ngrams(segments):
        if gram in _STOPWORDS:
            continue
        counter[gram] += 1

    return counter


def _top_dynamic_terms(text: str, top_n: int = _MAX_DYNAMIC_TERMS) -> list[str]:
    counter = _keyword_counter(text)
    if not counter:
        return []

    ranked = sorted(
        counter.items(),
        key=lambda item: (item[1] * min(len(item[0]), 6), item[1], len(item[0])),
        reverse=True,
    )
    return [term for term, _ in ranked[:top_n]]


def _build_dynamic_profile(query_text: str) -> dict[str, list[str]]:
    terms = _normalize_terms(_top_dynamic_terms(query_text, top_n=_MAX_DYNAMIC_PROFILE_TERMS))
    if not terms:
        return {}
    return {"dynamic_query": terms}


def _active_profiles_for_query(query_text: str) -> dict[str, list[str]]:
    lower = query_text.lower()
    profiles = _all_profiles()

    active: dict[str, list[str]] = {}
    for name, terms in profiles.items():
        if any(term in lower for term in terms):
            active[name] = terms
    return active


def _merge_profiles(*profile_dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for profile_dict in profile_dicts:
        for name, terms in profile_dict.items():
            key = _normalize_term(name)
            if not key:
                continue
            existing = merged.get(key, [])
            merged[key] = _normalize_terms(existing + [str(term) for term in terms])
    return merged


def _extract_tokens(text: str) -> set[str]:
    lower = text.lower()
    tokens: set[str] = set(re.findall(r"[a-zA-Z][a-zA-Z0-9\+\.#-]+", lower))

    segments = _extract_chinese_segments(text)
    for seg in segments:
        tokens.add(seg)
    tokens.update(_extract_chinese_ngrams(segments, min_n=2, max_n=3))

    for term in _all_profile_terms():
        if term in lower:
            tokens.add(term)

    return tokens


def _lexical_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0

    return overlap / (math.sqrt(len(query_tokens)) * math.sqrt(len(doc_tokens)))


def _profile_score(text: str, terms: list[str]) -> float:
    lower_text = text.lower()
    hits = sum(1 for term in terms if term.lower() in lower_text)
    return hits / max(1, len(terms))


def _profile_scores(text: str, profiles: dict[str, list[str]]) -> dict[str, float]:
    return {name: _profile_score(text, terms) for name, terms in profiles.items()}


def _extract_max_years(text: str) -> float | None:
    values: list[float] = []
    for match in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*年", text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0.5 <= value <= 40:
            values.append(value)
    if not values:
        return None
    return max(values)


def _weighted_candidate_score(query_text: str, candidate_text: str, base_score: float) -> float:
    if base_score <= 0:
        return 0.0

    query_terms = set(_normalize_terms(_top_dynamic_terms(query_text)))
    candidate_terms = set(_normalize_terms(_top_dynamic_terms(candidate_text)))

    profiles = _merge_profiles(_active_profiles_for_query(query_text), _build_dynamic_profile(query_text))
    query_profile = _profile_scores(query_text, profiles) if profiles else {}
    candidate_profile = _profile_scores(candidate_text, profiles) if profiles else {}

    term_coverage = 0.0
    if query_terms:
        term_coverage = len(query_terms & candidate_terms) / max(1, len(query_terms))

    score = base_score
    score *= 1.0 + 1.2 * term_coverage

    if query_profile:
        dominant_domain, dominant_value = max(query_profile.items(), key=lambda x: x[1])
        if dominant_value > 0:
            dominant_match = candidate_profile.get(dominant_domain, 0.0)
            score *= 1.0 + 0.8 * dominant_match

            off_domain = [name for name in query_profile if name != dominant_domain]
            off_penalty = sum(candidate_profile.get(name, 0.0) for name in off_domain)
            score *= max(0.65, 1.0 - 0.2 * off_penalty)

    required_years = _extract_max_years(query_text)
    candidate_years = _extract_max_years(candidate_text)
    if required_years is not None and candidate_years is not None:
        if candidate_years >= required_years:
            score *= 1.15
        elif candidate_years + 0.5 >= required_years:
            score *= 1.05
        else:
            score *= 0.78

    lower_query = query_text.lower()
    lower_candidate = candidate_text.lower()
    leadership_query = any(token in lower_query for token in ["负责人", "leader", "主管", "manager", "xpm", "主设"])
    if leadership_query:
        leadership_hits = sum(
            1 for token in ["负责人", "leader", "主管", "manager", "xpm", "主设", "统筹", "带团队"] if token in lower_candidate
        )
        if leadership_hits > 0:
            score *= 1.0 + min(0.2, 0.05 * leadership_hits)

    return score


class LexicalSearchService:
    def retrieve_candidates_for_job(self, job: Job, candidates: Iterable[Candidate], top_k: int) -> list[dict]:
        query_text = _job_text(job)
        query_tokens = _extract_tokens(query_text)

        scored: list[dict] = []
        for candidate in candidates:
            candidate_text = _candidate_text(candidate)
            doc_tokens = _extract_tokens(candidate_text)
            base_score = _lexical_score(query_tokens, doc_tokens)
            score = _weighted_candidate_score(query_text, candidate_text, base_score)
            if score > 0:
                scored.append({"candidate_id": candidate.id, "score": float(score)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def retrieve_jobs_for_candidate(self, candidate: Candidate, jobs: Iterable[Job], top_k: int) -> list[dict]:
        query_text = _candidate_text(candidate)
        query_tokens = _extract_tokens(query_text)

        scored: list[dict] = []
        for job in jobs:
            doc_tokens = _extract_tokens(_job_text(job))
            score = _lexical_score(query_tokens, doc_tokens)
            if score > 0:
                scored.append({"job_id": job.id, "score": float(score)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


lexical_search_service = LexicalSearchService()
