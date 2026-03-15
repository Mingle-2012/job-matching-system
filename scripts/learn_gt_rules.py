import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.parser import parse_text_from_file
from ingestion.skill_extractor import skill_extractor
from scripts.evaluate_cv_dataset import JobGroundTruth, extract_candidate_name, normalize_name, read_cv_gt
from services.llm_client import create_openai_client


@dataclass
class ResumeRecord:
    name: str
    text: str
    path: Path


def _safe_json_load(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    if not content:
        return {}

    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        snippet = content[start : end + 1]
        try:
            value = json.loads(snippet)
            if isinstance(value, dict):
                return value
        except Exception:
            return {}

    return {}


def _call_llm_json(client: Any, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    if client is None:
        return {}

    try:
        from config.settings import get_settings

        settings = get_settings()
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return _safe_json_load(content)
    except Exception:
        return {}


def _slice_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_weights(raw: dict[str, Any]) -> dict[str, float]:
    defaults = {
        "skill_weight": 0.4,
        "vector_weight": 0.2,
        "role_weight": 0.2,
        "experience_weight": 0.1,
        "domain_weight": 0.1,
    }

    parsed: dict[str, float] = {}
    for key, default_value in defaults.items():
        try:
            parsed[key] = max(0.0, float(raw.get(key, default_value)))
        except Exception:
            parsed[key] = default_value

    total = sum(parsed.values())
    if total <= 0:
        return defaults

    return {key: value / total for key, value in parsed.items()}


def _heuristic_analysis(job_text: str) -> dict[str, Any]:
    profile = skill_extractor.extract_profile(job_text, allow_llm=False)
    return {
        "core_skills": profile.skills[:12],
        "tools": profile.tools[:8],
        "domain": profile.domain[:8],
        "seniority": "mid",
        "experience": [],
        "hidden_rules": [],
    }


def load_all_resumes(dataset_dir: Path) -> dict[str, ResumeRecord]:
    records: dict[str, ResumeRecord] = {}

    for role_dir in sorted([path for path in dataset_dir.iterdir() if path.is_dir()], key=lambda p: p.name):
        for resume_file in sorted(role_dir.iterdir(), key=lambda p: p.name):
            if not resume_file.is_file() or resume_file.suffix.lower() != ".pdf":
                continue

            text = parse_text_from_file(resume_file)
            name = normalize_name(extract_candidate_name(resume_file, resume_text=text))
            if not name:
                continue

            if name not in records:
                records[name] = ResumeRecord(name=name, text=text, path=resume_file)

    return records


def build_prompt_1(job_text: str, resume_texts: list[str]) -> str:
    blocks: list[str] = []
    for idx, text in enumerate(resume_texts, start=1):
        blocks.append(f"[Resume-{idx}]\n{_slice_text(text, 1400)}")

    joined_resumes = "\n\n".join(blocks)
    return f"""
You are an expert recruiter.

We have a job and a set of ground-truth matched resumes.

Your task:

Analyze why these resumes are considered good matches.

Job description:
{_slice_text(job_text, 5000)}

Ground truth resumes:
{joined_resumes[:22000]}

Please summarize:

1. Required core skills
2. Required tools
3. Required experience type
4. Required seniority level
5. Required domain
6. Soft skills if any
7. Hidden requirements not explicitly written

Return JSON:

{{
  "core_skills": [],
  "tools": [],
  "domain": [],
  "seniority": "",
  "experience": [],
  "hidden_rules": []
}}
""".strip()


def build_prompt_2(job_text: str, resume_text: str, candidate_name: str) -> str:
    return f"""
For each resume, explain why it matches the job.

Important: this resume is already a ground-truth positive sample.
Focus on positive evidence and only mention minor risks if necessary.

Job description:
{_slice_text(job_text, 5000)}

Candidate name:
{candidate_name}

Resume text:
{_slice_text(resume_text, 4500)}

Return JSON:
{{
  "name": "",
  "match_reason": "",
  "skills": [],
  "role": "",
  "seniority": "",
  "domain": ""
}}
""".strip()


def build_prompt_3(job_text: str, analysis: dict[str, Any]) -> str:
    return f"""
Based on job and GT analysis, generate scoring rules for ranking.

Job description:
{_slice_text(job_text, 5000)}

GT analysis:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Return weights in JSON:
{{
  "skill_weight": 0.4,
  "vector_weight": 0.2,
  "role_weight": 0.2,
  "experience_weight": 0.1,
  "domain_weight": 0.1
}}
""".strip()


def learn_rule_for_job(
    client: Any,
    row: JobGroundTruth,
    resume_records: dict[str, ResumeRecord],
    include_labels: bool,
) -> dict[str, Any]:
    gt_names = list(dict.fromkeys([normalize_name(name) for name in row.screened_names if normalize_name(name)]))

    matched_records = [resume_records[name] for name in gt_names if name in resume_records]
    resume_texts = [record.text for record in matched_records]

    analysis_prompt = build_prompt_1(row.job_responsibility, resume_texts)
    analysis = _call_llm_json(
        client=client,
        system_prompt="You are an expert recruiter. Return strict JSON only.",
        user_prompt=analysis_prompt,
    )
    if not analysis:
        analysis = _heuristic_analysis(row.job_responsibility)

    normalized_analysis = {
        "core_skills": _to_string_list(analysis.get("core_skills")),
        "tools": _to_string_list(analysis.get("tools")),
        "domain": _to_string_list(analysis.get("domain")),
        "seniority": str(analysis.get("seniority", "")).strip().lower(),
        "experience": _to_string_list(analysis.get("experience")),
        "hidden_rules": _to_string_list(analysis.get("hidden_rules")),
    }

    weight_prompt = build_prompt_3(row.job_responsibility, normalized_analysis)
    weight_json = _call_llm_json(
        client=client,
        system_prompt="You design ranking systems. Return strict JSON only.",
        user_prompt=weight_prompt,
    )
    weights = _normalize_weights(weight_json)

    labels: list[dict[str, Any]] = []
    if include_labels:
        for record in matched_records:
            label_prompt = build_prompt_2(row.job_responsibility, record.text, record.name)
            label = _call_llm_json(
                client=client,
                system_prompt="You are an expert recruiter. Return strict JSON only.",
                user_prompt=label_prompt,
            )
            if not label:
                label = {
                    "name": record.name,
                    "match_reason": "Matched by GT list.",
                    "skills": [],
                    "role": "",
                    "seniority": "",
                    "domain": "",
                }
            labels.append(label)

    return {
        "job_name": row.job_name,
        "job_text": row.job_responsibility,
        "gt_names": gt_names,
        "analysis": normalized_analysis,
        "weights": weights,
        "resume_labels": labels,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Learn GT-driven rules with LLM for ranking")
    parser.add_argument("--dataset-dir", default=str(repo_root / "dataset"), help="Dataset directory path")
    parser.add_argument(
        "--output",
        default=str(repo_root / "dataset" / "gt_learned_rules.json"),
        help="Output JSON path for learned rules",
    )
    parser.add_argument("--include-labels", action="store_true", help="Generate per-resume labels (Prompt 2)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    csv_path = dataset_dir / "cv_gt.csv"
    if not dataset_dir.exists() or not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_dir}")

    gt_rows = read_cv_gt(csv_path)
    resume_records = load_all_resumes(dataset_dir)
    client = create_openai_client()

    learned_jobs: list[dict[str, Any]] = []
    for row in gt_rows:
        learned_jobs.append(
            learn_rule_for_job(
                client=client,
                row=row,
                resume_records=resume_records,
                include_labels=args.include_labels,
            )
        )

    output = {
        "meta": {
            "job_count": len(learned_jobs),
            "resume_count": len(resume_records),
            "include_labels": bool(args.include_labels),
        },
        "jobs": learned_jobs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"learned jobs: {len(learned_jobs)}")
    print(f"resume records: {len(resume_records)}")
    print(f"rules written to: {output_path}")


if __name__ == "__main__":
    main()
