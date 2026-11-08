"""Step 2: inspect WCB batches and check loss gradients with artificial embeddings.

This command does not load, fine-tune, or evaluate BGE-M3.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
from torch import nn

from .prepare import PROJECT_ROOT
from .triplet import hardest_indices, make_loss, validate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "data/processed/wcb_time_5768/train.parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5768)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        parser.error("batch-size must be an even number >=4 for this two-class example")
    rows = pq.read_table(args.data).to_pylist()
    rng = random.Random(args.seed)
    batch = []
    for label in (0, 1):
        candidates = [r for r in rows if r["label"] == label]
        if len(candidates) < args.batch_size // 2:
            parser.error("Not enough unique sentences for requested batch size")
        batch.extend(rng.sample(candidates, args.batch_size // 2))
    rng.shuffle(batch)
    labels = torch.tensor([r["label"] for r in batch], dtype=torch.long)
    validate_batch(labels)

    # Artificial vectors test loss plumbing and autograd, not text semantics.
    torch.manual_seed(args.seed)
    embeddings = torch.randn(len(batch), 16, requires_grad=True)
    loss_fn = make_loss(nn.Identity(), margin=args.margin)
    loss = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    loss.backward()
    if not torch.isfinite(loss) or embeddings.grad is None or not torch.isfinite(embeddings.grad).all():
        raise RuntimeError("Loss or embedding gradients are not finite")
    if loss.item() <= 0 or embeddings.grad.norm().item() == 0:
        raise RuntimeError("Smoke check did not exercise an active triplet")
    positive_indices, negative_indices = hardest_indices(embeddings.detach(), labels)
    assert torch.all(labels[positive_indices] == labels)
    assert torch.all(labels[negative_indices] != labels)
    assert torch.all(positive_indices != torch.arange(len(batch)))

    summary = {
        "task": "time_label",
        "batch_label_counts": dict(Counter(r["time_label"] for r in batch)),
        "embedding_source": "artificial_random_vectors_for_smoke_check_only",
        "loss": loss.item(),
        "gradient_norm": embeddings.grad.norm().item(),
        "bge_m3_loaded": False,
        "model_training_performed": False,
        "batch": [{"index": i, "sentence_id": r["sentence_id"],
                   "label": r["time_label"], "sentence": r["sentence"],
                   "positive_index": positive_indices[i].item(),
                   "negative_index": negative_indices[i].item()}
                  for i, r in enumerate(batch)],
    }
    target = args.data.parent / "batch_smoke_check.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "batch"}, indent=2))
    print(f"Batch and triplets: {target}")


if __name__ == "__main__":
    main()
