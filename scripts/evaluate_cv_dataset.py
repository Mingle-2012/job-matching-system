import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ingestion.parser import parse_text_from_file


@dataclass
class JobGroundTruth:
    index: int
    job_name: str
    job_responsibility: str
    screened_names: list[str]
    hired_names: list[str]


LOCATION_TOKENS = {
    "上海",
    "西安",
    "东莞",
    "深圳",
    "长沙",
    "桂林",
    "惠州",
    "北京",
    "广州",
    "南京",
    "杭州",
}

NOISE_NAME_TOKENS = {
    "专科",
    "本科",
    "硕士",
    "博士",
    "大专",
    "中专",
    "研究生",
    "结构",
    "射频",
    "工程师",
    "开发",
    "设计",
    "产品",
    "应用",
    "通讯",
    "底软",
    "通信",
}


def normalize_name(name: str) -> str:
    cleaned = str(name).replace("\u3000", " ").strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def is_valid_person_name(name: str) -> bool:
    normalized = normalize_name(name)
    if not normalized:
        return False
    if normalized in LOCATION_TOKENS or normalized in NOISE_NAME_TOKENS:
        return False
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", normalized):
        return False
    return True


def split_multiline_names(raw: str) -> list[str]:
    if not raw:
        return []
    values = [normalize_name(line) for line in str(raw).splitlines()]
    return [value for value in values if value]


def read_cv_gt(csv_path: Path) -> list[JobGroundTruth]:
    rows: list[JobGroundTruth] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                JobGroundTruth(
                    index=int(row["序号"]),
                    job_name=row["岗位名称"].strip(),
                    job_responsibility=row["岗位职责"].strip(),
                    screened_names=split_multiline_names(row.get("简历初筛通过人员", "")),
                    hired_names=split_multiline_names(row.get("入职人员", "")),
                )
            )
    return rows


def extract_name_from_resume_text(text: str) -> str | None:
    if not text:
        return None

    patterns = [
        r"(?:姓\s*名|姓名)\s*[:：]\s*([\u4e00-\u9fff]{2,4})",
        r"^([\u4e00-\u9fff]{2,4})\s+(?:简历|个人简历)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            continue
        candidate = normalize_name(match.group(1))
        if is_valid_person_name(candidate):
            return candidate

    return None


def extract_candidate_name(file_path: Path, resume_text: str = "") -> str:
    stem = file_path.stem.replace("docx..", "")
    parts = [normalize_name(part) for part in re.split(r"[-_]", stem) if normalize_name(part)]

    for part in reversed(parts):
        if is_valid_person_name(part):
            return part

    extracted = extract_name_from_resume_text(resume_text)
    if extracted:
        return extracted

    chinese_fragments = re.findall(r"[\u4e00-\u9fff]{2,4}", stem)
    for fragment in reversed(chinese_fragments):
        candidate = normalize_name(fragment)
        if is_valid_person_name(candidate):
            return candidate

    return stem[:32]


def build_candidate_payload(name: str, text: str) -> dict[str, Any]:
    if not text:
        text = f"候选人：{name}"

    return {
        "name": name,
        "job_status": "open_to_work",
        "resume_summary": text[:6000],
        "project_experience": text[6000:12000],
        "achievements": text[12000:18000],
    }


def ingest_candidates(dataset_dir: Path, api_base: str, timeout: int, verbose: bool) -> dict[int, str]:
    id_to_name: dict[int, str] = {}
    seen_names: set[str] = set()

    for role_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
        for resume_file in sorted(role_dir.iterdir(), key=lambda x: x.name):
            if not resume_file.is_file() or resume_file.suffix.lower() != ".pdf":
                continue

            resume_text = parse_text_from_file(resume_file)
            candidate_name = extract_candidate_name(resume_file, resume_text=resume_text)
            normalized = normalize_name(candidate_name)
            if not normalized or not is_valid_person_name(normalized) or normalized in seen_names:
                continue

            payload = build_candidate_payload(normalized, resume_text)
            resp = requests.post(
                f"{api_base}/ingest/candidate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()

            candidate_id = int(resp.json()["candidate_id"])
            id_to_name[candidate_id] = normalized
            seen_names.add(normalized)

            if verbose:
                print(f"[candidate] {normalized} -> id={candidate_id}")

    return id_to_name


def ingest_jobs(rows: list[JobGroundTruth], api_base: str, timeout: int, verbose: bool) -> dict[str, int]:
    job_name_to_id: dict[str, int] = {}
    for row in rows:
        payload = {
            "title": row.job_name,
            "status": "open",
            "job_description": row.job_responsibility,
            "responsibilities": row.job_responsibility,
            "preferred_qualifications": row.job_responsibility,
        }
        resp = requests.post(f"{api_base}/ingest/job", json=payload, timeout=timeout)
        resp.raise_for_status()
        job_id = int(resp.json()["job_id"])
        job_name_to_id[row.job_name] = job_id
        if verbose:
            print(f"[job] {row.job_name} -> id={job_id}")
    return job_name_to_id


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def resolve_candidate_name(
    candidate_id: int,
    id_to_name: dict[int, str],
    api_base: str,
    timeout: int,
    cache: dict[int, str | None],
) -> str | None:
    if candidate_id in id_to_name:
        return normalize_name(id_to_name[candidate_id])

    if candidate_id in cache:
        return cache[candidate_id]

    try:
        resp = requests.get(f"{api_base}/candidates/{candidate_id}/name", timeout=timeout)
        if resp.status_code == 200:
            raw_name = str(resp.json().get("name", "")).strip()
            normalized = normalize_name(raw_name)
            cache[candidate_id] = normalized if normalized else None
            return cache[candidate_id]
    except Exception:
        pass

    cache[candidate_id] = None
    return None


def jaccard_at_k(predicted: list[str], truth: set[str], k: int) -> float:
    pred_set = set(predicted[:k])
    union = pred_set.union(truth)
    if not union:
        return 1.0
    return len(pred_set.intersection(truth)) / len(union)


def precision_recall_f1_at_k(predicted: list[str], truth: set[str], k: int) -> tuple[float, float, float]:
    pred_k = predicted[:k]
    if not pred_k:
        return 0.0, 0.0, 0.0

    tp = len(set(pred_k).intersection(truth))
    precision = tp / len(pred_k)
    recall = tp / max(1, len(truth))
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def ap_at_k(predicted: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    seen_relevant: set[str] = set()

    for rank, name in enumerate(predicted[:k], start=1):
        if name in truth and name not in seen_relevant:
            seen_relevant.add(name)
            hit_count += 1
            precision_sum += hit_count / rank

    denom = min(len(truth), k)
    if denom == 0:
        return 0.0
    return precision_sum / denom


def ndcg_at_k(predicted: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0

    dcg = 0.0
    for i, name in enumerate(predicted[:k]):
        rel = 1.0 if name in truth else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate(
    rows: list[JobGroundTruth],
    api_base: str,
    timeout: int,
    top_k: int,
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    top_k = max(1, int(top_k))
    all_dataset_names = set(id_to_name.values())
    per_job: list[dict[str, Any]] = []
    candidate_name_cache: dict[int, str | None] = {}

    job_name_to_id = ingest_jobs(rows, api_base=api_base, timeout=timeout, verbose=False)

    for row in rows:
        job_id = job_name_to_id[row.job_name]
        resp = requests.post(
            f"{api_base}/search/candidates",
            json={"job_id": job_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_results = resp.json()

        predicted_list: list[str] = []
        for item in raw_results:
            candidate_id = int(item.get("candidate_id", 0))
            if candidate_id <= 0:
                continue
            resolved_name = resolve_candidate_name(
                candidate_id=candidate_id,
                id_to_name=id_to_name,
                api_base=api_base,
                timeout=timeout,
                cache=candidate_name_cache,
            )
            if resolved_name:
                predicted_list.append(resolved_name)

        predicted_names = unique_preserve_order(predicted_list)

        truth = set(row.screened_names)
        hired_truth = set(row.hired_names)

        dynamic_k = max(1, len(truth))
        metric_k = top_k

        coverage = len(truth.intersection(all_dataset_names)) / max(1, len(truth))
        jaccard_dynamic = jaccard_at_k(predicted_names, truth, dynamic_k)
        precision_k, recall_k, f1_k = precision_recall_f1_at_k(predicted_names, truth, metric_k)
        map_k = ap_at_k(predicted_names, truth, metric_k)
        ndcg_k = ndcg_at_k(predicted_names, truth, metric_k)

        _, hired_recall_k, _ = precision_recall_f1_at_k(predicted_names, hired_truth, metric_k)

        per_job.append(
            {
                "job_name": row.job_name,
                "job_id": job_id,
                "gt_count": len(truth),
                "hired_count": len(hired_truth),
                "coverage_in_dataset": coverage,
                "jaccard_at_gt_count": jaccard_dynamic,
                f"precision@{metric_k}": precision_k,
                f"recall@{metric_k}": recall_k,
                f"f1@{metric_k}": f1_k,
                f"map@{metric_k}": map_k,
                f"ndcg@{metric_k}": ndcg_k,
                f"hired_recall@{metric_k}": hired_recall_k,
                "predicted_top_names": predicted_names[:metric_k],
                "gt_names": sorted(list(truth)),
            }
        )

    macro = {
        "avg_coverage_in_dataset": sum(item["coverage_in_dataset"] for item in per_job) / max(1, len(per_job)),
        "avg_jaccard_at_gt_count": sum(item["jaccard_at_gt_count"] for item in per_job) / max(1, len(per_job)),
        f"avg_precision@{top_k}": sum(item[f"precision@{top_k}"] for item in per_job) / max(1, len(per_job)),
        f"avg_recall@{top_k}": sum(item[f"recall@{top_k}"] for item in per_job) / max(1, len(per_job)),
        f"avg_f1@{top_k}": sum(item[f"f1@{top_k}"] for item in per_job) / max(1, len(per_job)),
        f"avg_map@{top_k}": sum(item[f"map@{top_k}"] for item in per_job) / max(1, len(per_job)),
        f"avg_ndcg@{top_k}": sum(item[f"ndcg@{top_k}"] for item in per_job) / max(1, len(per_job)),
        f"avg_hired_recall@{top_k}": sum(item[f"hired_recall@{top_k}"] for item in per_job) / max(1, len(per_job)),
    }

    return {
        "macro": macro,
        "per_job": per_job,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on cv_gt dataset")
    parser.add_argument("--api-base", default="http://localhost:8000", help="FastAPI base URL")
    parser.add_argument("--dataset-dir", default=str(repo_root / "dataset"), help="Dataset directory path")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for ranking metrics")
    parser.add_argument(
        "--output",
        default=str(repo_root / "dataset" / "eval_report.json"),
        help="Evaluation report output JSON path",
    )
    parser.add_argument("--verbose", action="store_true", help="Print ingestion details")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    csv_path = dataset_dir / "cv_gt.csv"

    if not dataset_dir.exists() or not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_dir}")

    try:
        requests.get(f"{args.api_base}/health", timeout=10).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach backend at {args.api_base}, start API service first"
        ) from exc

    gt_rows = read_cv_gt(csv_path)
    id_to_name = ingest_candidates(dataset_dir, api_base=args.api_base, timeout=args.timeout, verbose=args.verbose)

    report = evaluate(
        rows=gt_rows,
        api_base=args.api_base,
        timeout=args.timeout,
        top_k=args.top_k,
        id_to_name=id_to_name,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Macro Metrics ===")
    for key, value in report["macro"].items():
        print(f"{key}: {value:.4f}")

    print("\n=== Per Job Summary ===")
    for item in report["per_job"]:
        print(
            f"{item['job_name']} | Jaccard@|GT|={item['jaccard_at_gt_count']:.4f} "
            f"| Recall@{args.top_k}={item[f'recall@{args.top_k}']:.4f} "
            f"| nDCG@{args.top_k}={item[f'ndcg@{args.top_k}']:.4f}"
        )

    print(f"\nreport written to: {output_path}")


if __name__ == "__main__":
    main()
