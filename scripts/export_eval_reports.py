"""Export small aggregate results for Git without raw text or model artifacts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "runs/wcb-stable-all/pipeline_summary.json": "reports/wcb-stable-all.json",
    "runs/wcb-stable-all/paper_comparison_metrics.json": "reports/paper-comparison.json",
    "runs/cross-seed-evaluation/summary.json": "reports/cross-seed-evaluation.json",
}


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return value[len(str(ROOT)) + 1:]
    return value


def main():
    for source, destination in SOURCES.items():
        if not (ROOT / source).exists():
            print("Skipped missing report:", source)
            continue
        output = ROOT / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(portable(json.loads((ROOT / source).read_text())),
                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Exported:", destination)


if __name__ == "__main__":
    main()
