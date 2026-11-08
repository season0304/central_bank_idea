"""Stage two: freeze the triplet encoder and train a dense softmax classifier."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import time

import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

from .batching import check_split_overlap, load_rows
from .tasks import labels_for
from .train import (EventLog, atomic_json, encode_batch, resolve_path, restore_rng,
                    rng_state, sha256)


@dataclass
class ClassifierConfig:
    task: str = "time_label"
    encoder_path: str = "runs/wcb-4090-all/time_label/triplet/best_model"
    data_dir: str = "data/processed_4090/wcb_time_5768"
    epochs: int = 50
    batch_size: int = 256
    embedding_batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    selection_metric: str = "macro_f1"
    seed: int = 5768
    precision: str = "auto"
    max_seq_length: int = 256
    feature_scaling: str = "standard"


def config_from_dict(value):
    """Existing runs used unscaled features; retain their resume behavior."""
    return ClassifierConfig(**{**value, "feature_scaling": value.get("feature_scaling", "none")})


def fit_feature_transform(embeddings, mode):
    """Fit on training embeddings only, in float32 and before moving to the GPU."""
    if mode == "none":
        return {"mode": "none"}
    if mode != "standard":
        raise ValueError("feature-scaling must be standard or none")
    values = embeddings.float().cpu()
    return {"mode": mode, "mean": values.mean(0),
            "scale": values.std(0, correction=0).clamp_min(1e-6)}


def transform_features(embeddings, transform=None):
    """Old classifier artifacts have no transform and consume raw features."""
    if transform is None or transform["mode"] == "none":
        return embeddings
    if transform["mode"] != "standard":
        raise ValueError("Unknown classifier feature transform")
    return ((embeddings - transform["mean"].to(embeddings)) /
            transform["scale"].to(embeddings))


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def metrics(logits, labels, class_count):
    predictions = logits.argmax(dim=1)
    confusion = torch.bincount(labels * class_count + predictions,
                               minlength=class_count ** 2).reshape(class_count, class_count)
    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"loss": F.cross_entropy(logits, labels).item(),
            "accuracy": (predictions == labels).float().mean().item(),
            "macro_f1": f1.mean().item(), "confusion_matrix": confusion.tolist(),
            "per_class": [{"label": i, "precision": precision[i].item(),
                           "recall": recall[i].item(), "f1": f1[i].item(),
                           "support": int(confusion[i].sum())} for i in range(class_count)]}


def encoder_hashes(path):
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and p.suffix in {".json", ".safetensors", ".bin", ".txt"}}


@torch.inference_mode()
def encode_sentences(encoder, sentences, device, precision):
    """Use the same pooling, precision and float32 normalization in all stages."""
    features = encoder.preprocess(sentences)
    features = {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()
    with context:
        embeddings = encoder(features)["sentence_embedding"].float()
    return F.normalize(embeddings, p=2, dim=1).cpu()


def make_embeddings(config, output, splits, device):
    """Cache embeddings; cache keys include encoder/data hashes and tokenization length."""
    source = resolve_path(config.encoder_path)
    hashes = encoder_hashes(source)
    if not hashes:
        raise ValueError(f"No encoder files: {source}")
    keys = {split: {"encoder": hashes, "data": sha256(resolve_path(config.data_dir) / f"{split}.parquet"),
                    "max_seq_length": config.max_seq_length, "normalize": True,
                    "precision": config.precision} for split in splits}
    paths = {split: output / f"embeddings_{split}.pt" for split in splits}
    result, missing = {}, []
    for split in splits:
        if paths[split].exists():
            saved = torch.load(paths[split], map_location="cpu", weights_only=True)
            if saved["fingerprint"] != keys[split]:
                raise ValueError(f"Embedding inputs changed: {paths[split]}; use a new output directory")
            result[split] = saved
        else:
            missing.append(split)
    if missing:
        encoder = SentenceTransformer(str(source), device=str(device), local_files_only=True,
                                      model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
                                      config_kwargs={"reference_compile": False})
        encoder.max_seq_length = config.max_seq_length
        encoder.eval()
        encoder.requires_grad_(False)
        for split in missing:
            rows = splits[split]
            vectors = []
            for start in range(0, len(rows), config.embedding_batch_size):
                # Inference batches need not contain multiple classes.
                sentences = [row["sentence"] for row in rows[start:start + config.embedding_batch_size]]
                vectors.append(encode_sentences(encoder, sentences, device, config.precision))
            saved = {"fingerprint": keys[split], "embeddings": torch.cat(vectors),
                     "labels": torch.tensor([r["label"] for r in rows]),
                     "sentence_ids": [r["sentence_id"] for r in rows]}
            atomic_torch(paths[split], saved)
            result[split] = saved
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _train(config, output, device, *, resume=False, stop_after_steps=None):
    label_map = labels_for(config.task)
    for key in ("epochs", "batch_size", "embedding_batch_size", "max_seq_length"):
        if getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive")
    if config.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be nonnegative (0 disables it)")
    if config.feature_scaling not in {"standard", "none"}:
        raise ValueError("feature-scaling must be standard or none")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0 or not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0:
        raise ValueError("min-delta must be finite and nonnegative")
    if config.selection_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("selection-metric must be accuracy or macro_f1")
    if config.precision == "auto":
        config.precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if config.precision not in {"fp32", "bf16"} or (config.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported())):
        raise ValueError("BF16 requires a supported CUDA GPU")
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop-after-steps must be positive")
    config_file = output / "config.json"
    if config_file.exists():
        if not resume:
            raise ValueError("Output exists; use --resume")
        if asdict(config_from_dict(json.loads(config_file.read_text()))) != asdict(config):
            raise ValueError("Classifier configuration changed")
    else:
        atomic_json(config_file, asdict(config))
    data_dir = resolve_path(config.data_dir)
    rows = {s: load_rows(data_dir / f"{s}.parquet", s, None, config.seed, config.task) for s in ("train", "val")}
    check_split_overlap(rows["train"], rows["val"])
    cached = make_embeddings(config, output, rows, device)
    fingerprints = {s: value["fingerprint"] for s, value in cached.items()}
    checkpoint = output / "checkpoint.pt"
    state = None
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["fingerprints"] != fingerprints or asdict(config_from_dict(state["config"])) != asdict(config):
            raise ValueError("Classifier checkpoint inputs/config changed")
        feature_transform = state.get("feature_transform", {"mode": "none"})
        if feature_transform["mode"] != config.feature_scaling:
            raise ValueError("Classifier checkpoint feature transform changed or is missing")
    else:
        feature_transform = fit_feature_transform(cached["train"]["embeddings"], config.feature_scaling)
    train_x = transform_features(cached["train"]["embeddings"], feature_transform).to(device)
    val_x = transform_features(cached["val"]["embeddings"], feature_transform).to(device)
    train_y, val_y = cached["train"]["labels"].to(device), cached["val"]["labels"].to(device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    head = nn.Linear(train_x.shape[1], len(label_map)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    progress = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_score": None,
                "best_loss": None, "patience_score": None, "bad_evaluations": 0,
                "early_stopped": False}
    if state is not None:
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        progress.update(state["progress"])
        restore_rng(state["rng"], device)
        # A crash may occur after the atomic best-head export but before checkpointing.
        best_path = output / "best_classifier.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=True)["metrics"]
            score = best[config.selection_metric]
            if progress["best_score"] is None or score > progress["best_score"] or (score == progress["best_score"] and best["loss"] < progress["best_loss"]):
                progress["best_score"], progress["best_loss"] = score, best["loss"]
    requested = {"signal": None}
    def stop(signum, frame):
        requested["signal"] = signum
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    log = EventLog(output / "events.jsonl")
    session_steps = 0
    def save():
        atomic_torch(checkpoint, {"config": asdict(config), "fingerprints": fingerprints,
                                 "feature_transform": feature_transform,
                                 "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                                 "progress": dict(progress), "rng": rng_state(device)})
    def validate():
        with torch.no_grad():
            value = metrics(head(val_x).float(), val_y, len(label_map))
        score = value[config.selection_metric]
        if progress["best_score"] is None or score > progress["best_score"] or (score == progress["best_score"] and value["loss"] < progress["best_loss"]):
            progress["best_score"], progress["best_loss"] = score, value["loss"]
            atomic_torch(output / "best_classifier.pt", {
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "embedding_dimension": train_x.shape[1], "label_to_id": label_map,
                "feature_transform": feature_transform,
                "task": config.task, "metrics": value, "global_step": progress["global_step"]})
        if progress["patience_score"] is None or score > progress["patience_score"] + config.early_stopping_min_delta:
            progress["patience_score"], progress["bad_evaluations"] = score, 0
        else:
            progress["bad_evaluations"] += 1
        progress["early_stopped"] = bool(config.early_stopping_patience and
                                         progress["bad_evaluations"] >= config.early_stopping_patience)
        log.write("validation", **progress, **value)
    try:
        log.write("start", task=config.task, frozen_encoder=True, selection_metric=config.selection_metric,
                  feature_scaling=config.feature_scaling)
        if not checkpoint.exists():
            validate()
            save()
        while progress["epoch"] < config.epochs and not progress["early_stopped"] and not requested["signal"]:
            generator = torch.Generator().manual_seed(config.seed + progress["epoch"])
            batches = torch.randperm(len(train_y), generator=generator).split(config.batch_size)
            for i in range(progress["next_batch"], len(batches)):
                if requested["signal"]:
                    break
                indices = batches[i].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(train_x[indices]).float(), train_y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite classifier loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                progress["global_step"] += 1
                session_steps += 1
                progress["next_batch"] = i + 1
                if i + 1 == len(batches):
                    progress["epoch"] += 1
                    progress["next_batch"] = 0
                    validate()
                    save()
                elif progress["global_step"] % 100 == 0:
                    save()
                if stop_after_steps and session_steps >= stop_after_steps:
                    break
            if stop_after_steps and session_steps >= stop_after_steps:
                break
        save()
        status = "early_stopped" if progress["early_stopped"] else ("completed" if progress["epoch"] >= config.epochs else "stopped")
        best = torch.load(output / "best_classifier.pt", map_location="cpu", weights_only=True)
        summary = {"status": status, "task": config.task, **progress,
                   "best_validation": best["metrics"], "test_split_used": False,
                   "encoder_frozen": True, "selection_metric": config.selection_metric,
                   "feature_scaling": config.feature_scaling,
                   "train_label_counts": dict(Counter(r["label"] for r in rows["train"]))}
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)
        log.close()


def train(config, output, *, device_name="cuda", resume=False, stop_after_steps=None, cpu_threads=4):
    output = resolve_path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    with (output / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another classifier process is using {output}") from exc
        return _train(config, output, device, resume=resume, stop_after_steps=stop_after_steps)


def evaluate_test(output, device_name="cuda"):
    """Run after model selection; never use test values to choose checkpoints."""
    output = resolve_path(output)
    config = config_from_dict(json.loads((output / "config.json").read_text()))
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] not in {"completed", "early_stopped"}:
        raise ValueError("Finish classifier selection before test evaluation")
    signature = {"classifier": sha256(output / "best_classifier.pt"),
                 "test_data": sha256(resolve_path(config.data_dir) / "test.parquet"),
                 "encoder": encoder_hashes(resolve_path(config.encoder_path))}
    if (output / "test_metrics.json").exists():
        saved = json.loads((output / "test_metrics.json").read_text())
        if saved["fingerprint"] != signature:
            raise ValueError("Test result inputs changed")
        return saved
    train_rows = load_rows(resolve_path(config.data_dir) / "train.parquet", "train", None, config.seed, config.task)
    val_rows = load_rows(resolve_path(config.data_dir) / "val.parquet", "val", None, config.seed, config.task)
    test_rows = load_rows(resolve_path(config.data_dir) / "test.parquet", "test", None, config.seed, config.task)
    check_split_overlap(train_rows, test_rows)
    check_split_overlap(val_rows, test_rows)
    cached = make_embeddings(config, output, {"test": test_rows}, torch.device(device_name))["test"]
    artifact = torch.load(output / "best_classifier.pt", map_location="cpu", weights_only=True)
    head = nn.Linear(artifact["embedding_dimension"], len(artifact["label_to_id"]))
    head.load_state_dict(artifact["head"])
    with torch.no_grad():
        features = transform_features(cached["embeddings"], artifact.get("feature_transform"))
        value = metrics(head(features), cached["labels"], len(artifact["label_to_id"]))
    result = {"task": config.task, "fingerprint": signature, "label_to_id": artifact["label_to_id"],
              "test_rows": len(test_rows), **value}
    atomic_json(output / "test_metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    for name, value in asdict(ClassifierConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    output = resolve_path(args.pop("output"))
    runtime = {k: args.pop(k) for k in ("resume", "device", "cpu_threads", "stop_after_steps", "evaluate_test")}
    if runtime["resume"]:
        saved = asdict(config_from_dict(json.loads((output / "config.json").read_text())))
        if any(saved[k] != v for k, v in args.items()):
            parser.error("Cannot change classifier config during resume")
        config = ClassifierConfig(**saved)
    else:
        config = ClassifierConfig(**args)
    result = train(config, output, device_name=runtime["device"], resume=runtime["resume"],
                   stop_after_steps=runtime["stop_after_steps"], cpu_threads=runtime["cpu_threads"])
    if runtime["evaluate_test"] and result["status"] in {"completed", "early_stopped"}:
        evaluate_test(output, runtime["device"])
    return result


if __name__ == "__main__":
    main()
