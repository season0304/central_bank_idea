"""Audit cached train/validation embeddings without selecting models on test data."""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from .batching import BalancedBatchSampler
from .classifier import fit_feature_transform, metrics, transform_features
from .train import atomic_json, resolve_path
from .triplet import batch_geometry


def embedding_geometry(embeddings, labels, *, margin=.2, batch_size=32, seed=5768):
    x = F.normalize(embeddings.double(), dim=1)
    n = len(x)
    # Average over distinct pairs, without materializing an n-by-n matrix.
    mean = x.mean(0)
    mean_cosine = (n * mean.dot(mean) - x.square().sum(1).mean()) / (n - 1)
    sampler = BalancedBatchSampler(labels.tolist(), batch_size, seed + 2)
    totals = {}
    batches = sampler.batches(0)
    for indices in batches:
        for key, value in batch_geometry(x[indices].float(), labels[indices], margin).items():
            totals[key] = totals.get(key, 0.) + value / len(batches)
    return {"rows": n, "mean_pairwise_cosine": mean_cosine.item(),
            "mean_feature_std": x.std(0, correction=0).mean().item(),
            "fixed_balanced_batches": totals}


def linear_probe(train, val, mode, *, epochs=50, seed=5768):
    transform = fit_feature_transform(train["embeddings"], mode)
    x = transform_features(train["embeddings"], transform)
    v = transform_features(val["embeddings"], transform)
    y, z = train["labels"], val["labels"]
    torch.manual_seed(seed)
    classes = int(y.max()) + 1
    head = nn.Linear(x.shape[1], classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    with torch.no_grad():
        best = {**metrics(head(v), z, classes), "epoch": 0}
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        for indices in torch.randperm(len(y), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[indices]), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            value = metrics(head(v), z, classes)
        if (value["macro_f1"], -value["loss"]) > (best["macro_f1"], -best["loss"]):
            best = {**value, "epoch": epoch + 1}
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifiers", nargs="+", required=True, help="Existing classifier directories")
    parser.add_argument("--baseline", help="Classifier directory with cached original-GTE validation embeddings")
    parser.add_argument("--output", default="runs/training-audit/audit.json")
    parser.add_argument("--probe-epochs", type=int, default=50)
    args = parser.parse_args(argv)
    if args.probe_epochs < 1:
        parser.error("probe-epochs must be positive")
    torch.set_num_threads(4)
    baseline = None
    if args.baseline:
        baseline = torch.load(resolve_path(args.baseline) / "embeddings_val.pt", map_location="cpu", weights_only=True)
    report = {"test_split_used": False, "selection_metric": "validation macro_f1",
              "probe": {"epochs": args.probe_epochs, "batch_size": 256, "learning_rate": .001,
                        "weight_decay": .01, "seed": 5768, "standardization_fit_split": "train"},
              "classifiers": {}}
    for directory in args.classifiers:
        source = resolve_path(directory)
        config = json.loads((source / "config.json").read_text())
        cached = {split: torch.load(source / f"embeddings_{split}.pt", map_location="cpu", weights_only=True)
                  for split in ("train", "val")}
        train, val = cached["train"], cached["val"]
        entry = {"source": str(source), "task": config["task"], "config": config,
                 "fingerprints": {split: value["fingerprint"] for split, value in cached.items()},
                 "validation_geometry": embedding_geometry(val["embeddings"], val["labels"]),
                 "validation_probes": {mode: linear_probe(train, val, mode, epochs=args.probe_epochs)
                                       for mode in ("none", "standard")}}
        if baseline is not None:
            lookup = {sentence_id: index for index, sentence_id in enumerate(baseline["sentence_ids"])}
            pairs = [(i, lookup[sid]) for i, sid in enumerate(val["sentence_ids"]) if sid in lookup]
            a, b = zip(*pairs)
            labels = val["labels"][list(a)]
            entry["same_sentences_comparison"] = {
                "matched_validation_rows": len(pairs), "baseline_source": args.baseline,
                "original_gte": embedding_geometry(baseline["embeddings"][list(b)], labels),
                "triplet_gte": embedding_geometry(val["embeddings"][list(a)], labels)}
        report["classifiers"][config["task"]] = entry
        print(json.dumps({"task": config["task"], "validation_macro_f1": {
            mode: value["macro_f1"] for mode, value in entry["validation_probes"].items()}}), flush=True)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print("Audit saved:", output)


if __name__ == "__main__":
    main()
