"""Predict WCB labels with portable triplet-encoder + dense-head artifacts."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

from .train import resolve_path
from .classifier import encode_sentences, transform_features


def predict(task_dir, sentences, device="cpu"):
    if not sentences:
        return []
    task_dir = Path(task_dir)
    artifact = torch.load(task_dir / "classifier" / "best_classifier.pt", map_location="cpu", weights_only=True)
    config = json.loads((task_dir / "classifier" / "config.json").read_text())
    encoder = SentenceTransformer(str(task_dir / "triplet" / "best_model"), device=device,
                                  local_files_only=True, model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
                                  config_kwargs={"reference_compile": False})
    encoder.max_seq_length = config["max_seq_length"]
    encoder.eval()
    head = nn.Linear(artifact["embedding_dimension"], len(artifact["label_to_id"]))
    head.load_state_dict(artifact["head"])
    head.eval()
    with torch.inference_mode():
        batch_size = config.get("embedding_batch_size", 64)
        vectors = torch.cat([encode_sentences(encoder, sentences[start:start + batch_size],
                                             torch.device(device), config["precision"])
                             for start in range(0, len(sentences), batch_size)])
        features = transform_features(vectors, artifact.get("feature_transform"))
        probabilities = F.softmax(head(features), dim=1)
    labels = sorted(artifact["label_to_id"], key=artifact["label_to_id"].get)
    return [{"sentence": sentence, "label": labels[int(scores.argmax())],
             "scores": {label: scores[i].item() for i, label in enumerate(labels)}}
            for sentence, scores in zip(sentences, probabilities)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/wcb-4090-all")
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = resolve_path(args.run)
    tasks = json.loads((root / "pipeline_config.json").read_text())["tasks"]
    result = {task: predict(root / task, args.text, args.device) for task in tasks}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
