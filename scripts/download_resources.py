"""Download source resources excluded from Git, without training any model."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Alibaba-NLP/gte-modernbert-base"
MODEL_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
DATASET = "gtfintechlab/all_annotated_sentences_25000"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", choices=["5768", "78516", "944601"], default=["5768"])
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    api = HfApi()
    model_revision = api.model_info(MODEL, revision=args.model_revision).sha
    data_revision = api.dataset_info(DATASET, revision=args.dataset_revision).sha
    snapshot_download(MODEL, revision=model_revision, local_dir=ROOT / "models/gte-modernbert-base",
        allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "1_Pooling/*"])
    patterns = [f"{seed}/{split}-00000-of-00001.parquet"
                for seed in args.seeds for split in ("train", "val", "test")]
    snapshot_download(DATASET, repo_type="dataset", revision=data_revision,
        local_dir=ROOT / "data/world_central_banks/all_annotated_sentences_25000",
        allow_patterns=[*patterns, "README.md"])
    manifest = {"model": {"repo_id": MODEL, "revision": model_revision},
                "dataset": {"repo_id": DATASET, "revision": data_revision, "seeds": args.seeds}}
    path = ROOT / "data/resource_downloads.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Resources downloaded; no training performed.")
    print("Pinned revisions:", path)


if __name__ == "__main__":
    main()
