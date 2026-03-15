import json
import re
from typing import Any, List

from config.settings import get_settings
from services.llm_client import create_openai_client

settings = get_settings()

_DOMAIN_KEYWORD_GROUPS = {
    "结构设计": ["结构", "xpm", "主设", "se", "整机", "机壳", "模具", "公差"],
    "MBB与驱动": ["mbb", "cpe", "网关", "驱动", "phy", "gmac", "pci", "pcie", "linux"],
    "射频与天线": ["射频", "rf", "天线", "4g", "5g", "高通", "mtk", "展锐", "3gpp", "lte"],
}


def _extract_english_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9\+\.#-]+", text.lower()))


def _extract_chinese_ngrams(text: str) -> set[str]:
    grams: set[str] = set()
    segments = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for seg in segments:
        length = len(seg)
        for n in (2, 3):
            if length < n:
                continue
            for i in range(0, length - n + 1):
                grams.add(seg[i : i + n])
    return grams


def _extract_domain_hits(text: str) -> set[str]:
    lower_text = text.lower()
    hits: set[str] = set()
    for group, keywords in _DOMAIN_KEYWORD_GROUPS.items():
        if any(keyword.lower() in lower_text for keyword in keywords):
            hits.add(group)
    return hits


class LLMReranker:
    def __init__(self) -> None:
        self.client = create_openai_client()
        self.llm_enabled = self.client is not None

    def evaluate_pair(self, candidate_text: str, job_text: str, context: str = "") -> dict[str, Any]:
        if self.llm_enabled and self.client:
            result = self._evaluate_with_llm(candidate_text, job_text, context=context)
            if result:
                return result
        return self._evaluate_with_heuristic(candidate_text, job_text, context=context)

    def _evaluate_with_llm(self, candidate_text: str, job_text: str, context: str = "") -> dict[str, Any] | None:
        context_block = f"\nRanking rules and hybrid signals:\n{context[:2200]}\n" if context else ""
        prompt = f"""
Evaluate the match between the following candidate and job.

Candidate Resume:
{candidate_text[:8000]}

Job Description:
{job_text[:8000]}
{context_block}

If ranking rules are provided above, you must follow them when scoring.

Evaluate:
1. Skill match
2. Experience relevance
3. Seniority alignment
4. Soft skills

Return JSON:
{{
  "match_score": 0-100,
  "reason": "..."
}}
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a recruiting expert and strict evaluator."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            score = int(max(0, min(100, int(data.get("match_score", 0)))))
            reason = str(data.get("reason", ""))
            return {"match_score": score, "reason": reason}
        except Exception:
            self.llm_enabled = False
            return None

    def _evaluate_with_heuristic(self, candidate_text: str, job_text: str, context: str = "") -> dict[str, Any]:
        candidate_tokens = _extract_english_tokens(candidate_text).union(_extract_chinese_ngrams(candidate_text))
        job_tokens = _extract_english_tokens(job_text).union(_extract_chinese_ngrams(job_text))

        if not candidate_tokens or not job_tokens:
            return {"match_score": 0, "reason": "Insufficient text for evaluation."}

        overlap = candidate_tokens & job_tokens
        token_ratio = len(overlap) / max(1, len(job_tokens))

        candidate_groups = _extract_domain_hits(candidate_text)
        job_groups = _extract_domain_hits(job_text)
        group_overlap = candidate_groups & job_groups
        group_ratio = len(group_overlap) / max(1, len(job_groups))

        score = int(min(100, 20 + token_ratio * 70 + group_ratio * 25))

        top_overlap = sorted([token for token in overlap if len(token) > 1])[:8]
        matched_groups = sorted(list(group_overlap))

        reason_parts = []
        if matched_groups:
            reason_parts.append("Matched domain areas: " + ", ".join(matched_groups))
        if top_overlap:
            reason_parts.append("Keyword overlap: " + ", ".join(top_overlap))
        if not reason_parts:
            reason_parts.append("Limited overlap across domain terms and experience text.")

        if context:
            reason_parts.append(f"Hybrid context: {context[:300]}")

        reason = " | ".join(reason_parts)
        return {"match_score": score, "reason": reason}

    def rerank_candidates_for_job(self, job_text: str, pairs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        llm_weight = max(0.0, min(1.0, settings.llm_rerank_weight))
        if llm_weight <= 0:
            for pair in pairs:
                prior_score = float(pair.get("hybrid_score", 0.0))
                results.append(
                    {
                        "candidate_id": pair["candidate_id"],
                        "score": prior_score,
                        "reason": f"hybrid_only | {pair.get('hybrid_reason', '')}",
                    }
                )
            return sorted(results, key=lambda x: x["score"], reverse=True)

        for pair in pairs:
            eval_result = self.evaluate_pair(pair["candidate_text"], job_text, context=pair.get("signal_prompt", ""))
            llm_score = eval_result["match_score"] / 100.0
            prior_score = float(pair.get("hybrid_score", 0.0))
            final_score = llm_weight * llm_score + (1.0 - llm_weight) * prior_score
            results.append(
                {
                    "candidate_id": pair["candidate_id"],
                    "score": final_score,
                    "reason": f"{eval_result['reason']} | hybrid_pre={prior_score:.3f}",
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def rerank_jobs_for_candidate(self, candidate_text: str, pairs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        llm_weight = max(0.0, min(1.0, settings.llm_rerank_weight))
        if llm_weight <= 0:
            for pair in pairs:
                prior_score = float(pair.get("hybrid_score", 0.0))
                results.append(
                    {
                        "job_id": pair["job_id"],
                        "score": prior_score,
                        "reason": f"hybrid_only | {pair.get('hybrid_reason', '')}",
                    }
                )
            return sorted(results, key=lambda x: x["score"], reverse=True)

        for pair in pairs:
            eval_result = self.evaluate_pair(candidate_text, pair["job_text"], context=pair.get("signal_prompt", ""))
            llm_score = eval_result["match_score"] / 100.0
            prior_score = float(pair.get("hybrid_score", 0.0))
            final_score = llm_weight * llm_score + (1.0 - llm_weight) * prior_score
            results.append(
                {
                    "job_id": pair["job_id"],
                    "score": final_score,
                    "reason": f"{eval_result['reason']} | hybrid_pre={prior_score:.3f}",
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)


reranker = LLMReranker()
