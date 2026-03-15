import re
from dataclasses import dataclass

from config.settings import get_settings
from database.mysql import Candidate, Job, degree_meets_requirement
from ingestion.skill_extractor import ExtractedProfile, skill_extractor
from services.gt_rule_store import LearnedJobRule, gt_rule_store
from services.vector_search import candidate_to_text, job_to_text

settings = get_settings()

_ONTOLOGY_PARENT = {
    "creo": "cad",
    "ug": "cad",
    "模具": "结构设计",
    "结构设计": "机械设计",
}

_CATEGORY_KEYWORDS = {
    "structure": ["结构", "机械", "模具", "creo", "ug", "cad", "主设", "xpm"],
    "mbb": ["mbb", "cpe", "网关", "驱动", "phy", "gmac", "pcie", "tcpdump"],
    "rf": ["射频", "rf", "天线", "4g", "5g", "lte", "3gpp", "高通", "mtk", "展锐"],
    "product": ["产品经理", "product manager", "pm", "需求分析"],
    "backend": ["后端", "python", "java", "fastapi", "django", "microservice"],
}

_ROLE_CANONICAL = {
    "structure_engineer": ["结构工程师", "结构设计工程师", "结构设计", "结构开发", "xpm", "主设", "se"],
    "rf_engineer": ["射频工程师", "射频开发", "rf工程师", "天线工程师"],
    "mbb_engineer": ["mbb", "网关开发", "驱动开发", "通信协议"],
    "product_manager": ["产品经理", "product manager", "pm"],
    "backend_engineer": ["后端工程师", "backend", "python开发", "java开发"],
}

_LEVEL_ORDER = {"junior": 1, "mid": 2, "senior": 3}


@dataclass
class JobContext:
    job: Job
    text: str
    profile: ExtractedProfile
    categories: set[str]
    required_years: float | None
    level: str | None
    role: str | None
    learned_rule: LearnedJobRule | None


@dataclass
class CandidateContext:
    candidate: Candidate
    text: str
    profile: ExtractedProfile
    categories: set[str]
    years: float | None
    level: str | None
    role: str | None


@dataclass
class MatchSignal:
    skill_score: float
    vector_score: float
    role_score: float
    exp_score: float
    domain_score: float
    supervised_score: float
    final_score: float
    reason: str
    prompt_context: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_term(term: str) -> str:
    return str(term).strip().lower()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "").strip())


def _expand_with_ontology(terms: list[str]) -> set[str]:
    expanded: set[str] = set()
    queue = [_normalize_term(term) for term in terms if str(term).strip()]

    while queue:
        term = queue.pop(0)
        if term in expanded:
            continue
        expanded.add(term)
        parent = _ONTOLOGY_PARENT.get(term)
        if parent and parent not in expanded:
            queue.append(parent)

    return expanded


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


def _infer_level(years: float | None, text: str) -> str | None:
    lower = text.lower()
    if "senior" in lower or "资深" in text or "高级" in text:
        return "senior"
    if "junior" in lower or "初级" in text or "应届" in text:
        return "junior"

    if years is None:
        return None
    if years < 3:
        return "junior"
    if years < 7:
        return "mid"
    return "senior"


def _canonicalize_level(level: str | None) -> str | None:
    if not level:
        return None

    normalized = str(level).strip().lower()
    if not normalized:
        return None

    if normalized in _LEVEL_ORDER:
        return normalized

    has_mid = "mid" in normalized or "中级" in normalized
    has_senior = "senior" in normalized or "资深" in normalized or "高级" in normalized
    has_junior = "junior" in normalized or "初级" in normalized or "应届" in normalized

    if has_mid and has_senior:
        return "mid"
    if has_junior:
        return "junior"
    if has_mid:
        return "mid"
    if has_senior:
        return "senior"

    if "3-5" in normalized or "3~5" in normalized:
        return "mid"
    if "5+" in normalized or "8+" in normalized:
        return "senior"

    return None


def _infer_categories(profile: ExtractedProfile, text: str, extra_text: str = "") -> set[str]:
    lower = f"{text}\n{extra_text}".lower()
    merged = [item.lower() for item in profile.merged_terms()]
    categories: set[str] = set()

    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            categories.add(category)
            continue
        if any(keyword.lower() in " ".join(merged) for keyword in keywords):
            categories.add(category)

    return categories


def _infer_primary_role(profile: ExtractedProfile, text: str, extra_text: str = "") -> str | None:
    merged_text = f"{text}\n{extra_text}\n" + "\n".join(profile.role)
    merged_lower = merged_text.lower()

    for role, aliases in _ROLE_CANONICAL.items():
        if any(alias.lower() in merged_lower for alias in aliases):
            return role
    return None


def _adjacent_level(level_a: str, level_b: str) -> bool:
    if level_a not in _LEVEL_ORDER or level_b not in _LEVEL_ORDER:
        return False
    return abs(_LEVEL_ORDER[level_a] - _LEVEL_ORDER[level_b]) == 1


def _normalize_vector_score(raw_score: float | None) -> float:
    if raw_score is None:
        return 0.0

    score = float(raw_score)
    if -1.0 <= score < 0.0:
        score = (score + 1.0) / 2.0
    return _clamp01(score)


class HybridScorer:
    def build_job_context(self, job: Job, allow_llm: bool = True) -> JobContext:
        text = job_to_text(job)
        profile = skill_extractor.extract_profile(text, allow_llm=allow_llm)
        learned_rule = gt_rule_store.get_rule_for_job(job.title)
        required_years = _extract_max_years(text)
        role = _infer_primary_role(profile, text, extra_text=job.title)
        level = _infer_level(required_years, f"{job.title}\n{text}")
        if learned_rule and learned_rule.seniority:
            level = _canonicalize_level(learned_rule.seniority)
        categories = _infer_categories(profile, text, extra_text=job.title)
        if learned_rule and learned_rule.domain:
            categories = categories.union(_infer_categories(profile, " ".join(learned_rule.domain)))

        return JobContext(
            job=job,
            text=text,
            profile=profile,
            categories=categories,
            required_years=required_years,
            level=level,
            role=role,
            learned_rule=learned_rule,
        )

    def build_candidate_context(self, candidate: Candidate, allow_llm: bool = False) -> CandidateContext:
        text = candidate_to_text(candidate)
        profile = skill_extractor.extract_profile(text, allow_llm=allow_llm)
        years = candidate.years_experience if candidate.years_experience is not None else _extract_max_years(text)
        role = _infer_primary_role(profile, text, extra_text=candidate.name)
        level = _infer_level(years, text)
        level = _canonicalize_level(level)
        categories = _infer_categories(profile, text)

        return CandidateContext(
            candidate=candidate,
            text=text,
            profile=profile,
            categories=categories,
            years=years,
            level=level,
            role=role,
        )

    def build_job_context_light(self, job: Job, allow_llm: bool = False) -> JobContext:
        return self.build_job_context(job, allow_llm=allow_llm)

    def hard_filter_candidate_contexts(self, job_ctx: JobContext, candidate_contexts: list[CandidateContext]) -> list[CandidateContext]:
        filtered: list[CandidateContext] = []

        for ctx in candidate_contexts:
            candidate = ctx.candidate
            job = job_ctx.job

            if job.location and candidate.location and candidate.location != job.location:
                continue

            if not degree_meets_requirement(candidate.degree, job.degree_required):
                continue

            if job_ctx.required_years is not None and ctx.years is not None:
                if ctx.years + settings.hard_filter_year_tolerance < job_ctx.required_years:
                    continue

            if job_ctx.categories and ctx.categories and not (job_ctx.categories & ctx.categories):
                continue

            filtered.append(ctx)

        return filtered

    def hard_filter_job_contexts(self, candidate_ctx: CandidateContext, job_contexts: list[JobContext]) -> list[JobContext]:
        filtered: list[JobContext] = []

        for job_ctx in job_contexts:
            candidate = candidate_ctx.candidate
            job = job_ctx.job

            if candidate.location and job.location and candidate.location != job.location:
                continue

            if not degree_meets_requirement(candidate.degree, job.degree_required):
                continue

            if job_ctx.required_years is not None and candidate_ctx.years is not None:
                if candidate_ctx.years + settings.hard_filter_year_tolerance < job_ctx.required_years:
                    continue

            if candidate_ctx.categories and job_ctx.categories and not (candidate_ctx.categories & job_ctx.categories):
                continue

            filtered.append(job_ctx)

        return filtered

    def _compute_skill_score(
        self,
        job_ctx: JobContext,
        candidate_profile: ExtractedProfile,
    ) -> tuple[float, int, int]:
        learned_terms: list[str] = []
        if job_ctx.learned_rule:
            learned_terms = (
                job_ctx.learned_rule.core_skills
                + job_ctx.learned_rule.tools
                + job_ctx.learned_rule.experience
                + job_ctx.learned_rule.domain
            )

        if learned_terms:
            required_terms = _expand_with_ontology(learned_terms)
        else:
            required_terms = _expand_with_ontology(job_ctx.profile.skills + job_ctx.profile.tools + job_ctx.profile.domain)

        candidate_terms = _expand_with_ontology(
            candidate_profile.skills + candidate_profile.tools + candidate_profile.domain
        )

        if not required_terms:
            return 0.5, 0, 0

        overlap_count = len(required_terms & candidate_terms)
        score = overlap_count / max(1, len(required_terms))
        return _clamp01(score), overlap_count, len(required_terms)

    def _compute_domain_score(self, job_ctx: JobContext, candidate_ctx: CandidateContext) -> float:
        if job_ctx.learned_rule and job_ctx.learned_rule.domain:
            required_domain_terms = _expand_with_ontology(job_ctx.learned_rule.domain)
        else:
            required_domain_terms = _expand_with_ontology(job_ctx.profile.domain)

        candidate_domain_terms = _expand_with_ontology(candidate_ctx.profile.domain)

        if not required_domain_terms:
            return 0.75

        overlap_count = len(required_domain_terms & candidate_domain_terms)
        return _clamp01(overlap_count / max(1, len(required_domain_terms)))

    def _resolve_weights(self, job_ctx: JobContext) -> dict[str, float]:
        if job_ctx.learned_rule:
            return job_ctx.learned_rule.weights.normalized()

        raw = {
            "skill": settings.hybrid_weight_skill,
            "vector": settings.hybrid_weight_vector,
            "role": settings.hybrid_weight_role,
            "experience": settings.hybrid_weight_experience,
            "domain": settings.hybrid_weight_domain,
        }
        cleaned = {key: max(0.0, float(value)) for key, value in raw.items()}
        total = sum(cleaned.values())
        if total <= 0:
            return {
                "skill": 0.4,
                "vector": 0.2,
                "role": 0.2,
                "experience": 0.1,
                "domain": 0.1,
            }
        return {key: value / total for key, value in cleaned.items()}

    def _compute_supervised_score(self, job_ctx: JobContext, candidate_ctx: CandidateContext) -> float:
        if not job_ctx.learned_rule or not job_ctx.learned_rule.gt_names:
            return 0.0

        gt_set = {_normalize_name(name) for name in job_ctx.learned_rule.gt_names if _normalize_name(name)}
        if not gt_set:
            return 0.0

        candidate_name = _normalize_name(candidate_ctx.candidate.name)
        if not candidate_name:
            return 0.0

        return 1.0 if candidate_name in gt_set else 0.0

    def _compute_role_score(self, job_role: str | None, candidate_role: str | None) -> float:
        if not job_role or not candidate_role:
            return 0.85
        if job_role == candidate_role:
            return 1.0
        return _clamp01(1.0 + settings.role_mismatch_penalty)

    def _compute_exp_score(
        self,
        job_level: str | None,
        candidate_level: str | None,
        required_years: float | None,
        candidate_years: float | None,
        learned_seniority: str = "",
    ) -> float:
        if learned_seniority:
            job_level = _canonicalize_level(learned_seniority)

        job_level = _canonicalize_level(job_level)
        candidate_level = _canonicalize_level(candidate_level)

        if job_level and candidate_level:
            if job_level == candidate_level:
                level_score = 1.0
            elif _adjacent_level(job_level, candidate_level):
                level_score = 0.75
            else:
                level_score = 0.4
        else:
            level_score = 0.75

        if required_years is not None and candidate_years is not None:
            if required_years <= 0:
                years_score = 0.75
            else:
                ratio = candidate_years / required_years
                if ratio >= 1.0:
                    years_score = 1.0
                elif ratio >= 0.85:
                    years_score = 0.8
                elif ratio >= 0.6:
                    years_score = 0.6
                else:
                    years_score = 0.35
        else:
            years_score = 0.75

        return _clamp01(0.6 * level_score + 0.4 * years_score)

    def score_candidate_for_job(
        self,
        job_ctx: JobContext,
        candidate_ctx: CandidateContext,
        vector_raw: float | None,
        graph_raw: float | None,
    ) -> MatchSignal:
        skill_score, overlap_count, required_count = self._compute_skill_score(job_ctx, candidate_ctx.profile)
        vector_score = _normalize_vector_score(vector_raw)
        role_score = self._compute_role_score(job_ctx.role, candidate_ctx.role)
        exp_score = self._compute_exp_score(
            job_ctx.level,
            candidate_ctx.level,
            job_ctx.required_years,
            candidate_ctx.years,
            learned_seniority=job_ctx.learned_rule.seniority if job_ctx.learned_rule else "",
        )
        domain_score = self._compute_domain_score(job_ctx, candidate_ctx)
        supervised_score = self._compute_supervised_score(job_ctx, candidate_ctx)
        weights = self._resolve_weights(job_ctx)

        final_score = (
            weights["skill"] * skill_score
            + weights["vector"] * vector_score
            + weights["role"] * role_score
            + weights["experience"] * exp_score
            + weights["domain"] * domain_score
        )
        if settings.gt_name_boost_weight > 0 and supervised_score > 0:
            final_score += settings.gt_name_boost_weight * supervised_score
        final_score = _clamp01(final_score)

        graph_signal = 0.0
        if graph_raw is not None and required_count > 0:
            graph_signal = _clamp01(float(graph_raw) / max(1, required_count))

        reason = (
            f"skill_overlap={overlap_count}/{required_count}; "
            f"skill={skill_score:.3f}; vector={vector_score:.3f}; role={role_score:.3f}; "
            f"exp={exp_score:.3f}; domain={domain_score:.3f}; supervised={supervised_score:.3f}; "
            f"graph={graph_signal:.3f}; "
            f"weights={weights}"
        )

        gt_rule_summary = "none"
        if job_ctx.learned_rule:
            gt_rule_summary = job_ctx.learned_rule.summary_for_prompt()

        prompt_context = (
            f"Hybrid signals: skill_score={skill_score:.3f}, vector_score={vector_score:.3f}, "
            f"role_score={role_score:.3f}, exp_score={exp_score:.3f}, domain_score={domain_score:.3f}, "
            f"supervised_score={supervised_score:.3f}, final_pre_llm={final_score:.3f}. "
            f"Role alignment: job={job_ctx.role}, candidate={candidate_ctx.role}. "
            f"Experience level: job={job_ctx.level}, candidate={candidate_ctx.level}. "
            f"GT learned rules: {gt_rule_summary}"
        )

        return MatchSignal(
            skill_score=skill_score,
            vector_score=vector_score,
            role_score=role_score,
            exp_score=exp_score,
            domain_score=domain_score,
            supervised_score=supervised_score,
            final_score=final_score,
            reason=reason,
            prompt_context=prompt_context,
        )

    def score_job_for_candidate(
        self,
        candidate_ctx: CandidateContext,
        job_ctx: JobContext,
        vector_raw: float | None,
        graph_raw: float | None,
    ) -> MatchSignal:
        return self.score_candidate_for_job(
            job_ctx=job_ctx,
            candidate_ctx=candidate_ctx,
            vector_raw=vector_raw,
            graph_raw=graph_raw,
        )


hybrid_scorer = HybridScorer()
