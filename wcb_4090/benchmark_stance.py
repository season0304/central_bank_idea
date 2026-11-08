"""Compare the published WCB stance classifier with our saved model on identical test rows."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download

from .batching import load_rows
from .classifier import metrics, transform_features
from .tasks import labels_for
from .train import atomic_json, resolve_path, sha256


REPO = "gtfintechlab/model_WCB_stance_label"
REVISION = "2a8d527f70ea7c58976eb01a3570a91512164570"
MODEL_CARD = f"https://huggingface.co/{REPO}/blob/{REVISION}/README.md"
# The model card's ordering differs from our dataset's integer ordering.
PUBLISHED_LABELS = ("neutral", "hawkish", "dovish", "irrelevant")


def remap_logits(logits):
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("Published stance model must have exactly four output logits")
    order = sorted(labels_for("stance_label"), key=labels_for("stance_label").get)
    return logits[:, [PUBLISHED_LABELS.index(label) for label in order]]


def paired_outcomes(local, published, labels):
    a, b = local.eq(labels), published.eq(labels)
    return {"both_correct": int((a & b).sum()), "local_only_correct": int((a & ~b).sum()),
            "published_only_correct": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum())}


def local_predictions(directory, rows, test_hash):
    saved = json.loads((directory / "test_metrics.json").read_text())
    if saved["fingerprint"]["test_data"] != test_hash:
        raise ValueError("Local model metrics refer to a different test dataset")
    if saved["fingerprint"]["classifier"] != sha256(directory / "best_classifier.pt"):
        raise ValueError("Local classifier changed after its saved test evaluation")
    cache = torch.load(directory / "embeddings_test.pt", map_location="cpu", weights_only=True)
    if cache["fingerprint"]["data"] != test_hash:
        raise ValueError("Local model embedding cache refers to a different test dataset")
    if cache["sentence_ids"] != [row["sentence_id"] for row in rows]:
        raise ValueError("Local model test sentence ordering differs")
    labels = torch.tensor([row["label"] for row in rows])
    if not torch.equal(labels, cache["labels"]):
        raise ValueError("Local model cache labels differ from the test dataset")
    artifact = torch.load(directory / "best_classifier.pt", map_location="cpu", weights_only=True)
    if artifact["label_to_id"] != labels_for("stance_label"):
        raise ValueError("Local classifier label mapping differs")
    head = nn.Linear(artifact["embedding_dimension"], 4)
    head.load_state_dict(artifact["head"])
    with torch.inference_mode():
        logits = head(transform_features(cache["embeddings"], artifact.get("feature_transform")))
    computed = metrics(logits, labels, 4)
    for key in ("accuracy", "macro_f1"):
        if abs(computed[key] - saved[key]) > 1e-6:
            raise ValueError("Local model's cached predictions do not reproduce its saved metrics")
    return computed, logits.argmax(1), labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/model_WCB_stance_label")
    parser.add_argument("--local-classifier", default="runs/wcb-stable-all/stance_label/classifier")
    parser.add_argument("--test-data", default="data/processed_4090/wcb_stance_5768/test.parquet")
    parser.add_argument("--output", default="runs/stance-benchmark/comparison.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--download", action="store_true", help="Fetch pinned published model files using existing Hub access")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch-size and max-length must be positive")
    torch.set_num_threads(4)
    model_dir, output = resolve_path(args.model_dir), resolve_path(args.output)
    test_data = resolve_path(args.test_data)
    rows = load_rows(test_data, "test", None, 5768, "stance_label")
    test_hash = sha256(test_data)
    local, local_pred, labels = local_predictions(resolve_path(args.local_classifier), rows, test_hash)
    report = {
        "status": "pending", "dataset": {"path": str(test_data), "sha256": test_hash,
            "rows": len(rows), "split": "test", "seed": "5768"},
        "label_to_id": labels_for("stance_label"),
        "local_model": {"path": args.local_classifier, **local},
        "published_model": {"repo_id": REPO, "revision": REVISION, "model_card": MODEL_CARD,
                            "published_id_to_label": dict(enumerate(PUBLISHED_LABELS))},
        "settings": {"max_length": args.max_length, "batch_size": args.batch_size, "device": args.device},
        "limitations": ["The published model card does not identify its training split seed; overlap with this 5768 test split cannot be ruled out."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.download:
            snapshot_download(REPO, revision=REVISION, local_dir=str(model_dir),
                allow_patterns=["config.json", "model.safetensors", "tokenizer*", "vocab.json",
                                "merges.txt", "special_tokens_map.json", "README.md"])
        if not (model_dir / "model.safetensors").exists() or not (model_dir / "config.json").exists():
            raise FileNotFoundError("Published model weights/config are missing; the local folder currently contains only model documentation")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float32)
    except Exception as exc:
        report["status"] = "blocked"
        report["published_model"]["error_type"] = type(exc).__name__
        report["published_model"]["error"] = str(exc)
        atomic_json(output, report)
        print(json.dumps({"status": "blocked", "reason": str(exc), "report": str(output)}, ensure_ascii=False))
        return 2
    if model.config.num_labels != 4:
        raise ValueError("Published stance model has a different number of classes")
    for i, expected in enumerate(PUBLISHED_LABELS):
        configured = str(model.config.id2label.get(i, "")).lower()
        if configured not in {expected, f"label_{i}"}:
            raise ValueError(f"Published model label {i} conflicts with its model card: {configured}")
    device = torch.device(args.device)
    model.to(device).eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = tokenizer([row["sentence"] for row in rows[start:start + args.batch_size]],
                              padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits.float().cpu()
            scores.append(remap_logits(logits))
            completed = min(start + args.batch_size, len(rows))
            if start == 0 or completed == len(rows) or completed % (25 * args.batch_size) == 0:
                print(json.dumps({"completed": completed, "total": len(rows)}), flush=True)
    logits = torch.cat(scores)
    published = metrics(logits, labels, 4)
    report["status"] = "completed"
    report["published_model"].update(published)
    report["published_model"]["weights_sha256"] = sha256(model_dir / "model.safetensors")
    report["delta_published_minus_local"] = {key: published[key] - local[key] for key in ("accuracy", "macro_f1")}
    report["paired_outcomes"] = paired_outcomes(local_pred, logits.argmax(1), labels)
    atomic_json(output, report)
    with (output.parent / "predictions.jsonl").open("w") as stream:
        for row, a, b, probabilities in zip(rows, local_pred, logits.argmax(1), logits.softmax(1)):
            stream.write(json.dumps({"sentence_id": row["sentence_id"], "true_label": row["label"],
                "local_prediction": int(a), "published_prediction": int(b),
                "published_probabilities": probabilities.tolist()}) + "\n")
    print(json.dumps({"report": str(output), "local": local, "published": published}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
