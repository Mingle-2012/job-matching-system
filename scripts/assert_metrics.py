import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert metric threshold on evaluation report")
    parser.add_argument("--report", required=True, help="Path to eval report json")
    parser.add_argument("--metric", default="recall@10", help="Metric key in per_job items")
    parser.add_argument("--macro-metric", default="avg_recall@10", help="Metric key in macro section")
    parser.add_argument("--threshold", type=float, default=0.6, help="Minimum acceptable value")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))

    failures: list[str] = []

    macro = report.get("macro", {})
    macro_value = float(macro.get(args.macro_metric, 0.0))
    if macro_value < args.threshold:
        failures.append(
            f"macro {args.macro_metric}={macro_value:.4f} < threshold {args.threshold:.4f}"
        )

    per_job = report.get("per_job", [])
    for item in per_job:
        job_name = str(item.get("job_name", "unknown"))
        value = float(item.get(args.metric, 0.0))
        if value < args.threshold:
            failures.append(
                f"job={job_name} {args.metric}={value:.4f} < threshold {args.threshold:.4f}"
            )

    if failures:
        print("metric assertion failed:")
        for message in failures:
            print("-", message)
        sys.exit(2)

    print(
        f"metric assertion passed: {args.metric} and {args.macro_metric} >= {args.threshold:.4f}"
    )


if __name__ == "__main__":
    main()
