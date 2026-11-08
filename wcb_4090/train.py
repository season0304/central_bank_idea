"""Fine-tune a local encoder on WCB labels with reference-preserving triplets.

This is the embedding-training stage, not training the downstream classifier.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import time
import uuid

import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
import sentence_transformers
import transformers

from .batching import BalancedBatchSampler, check_split_overlap, load_rows
from .prepare import LABEL_TO_ID, PROJECT_ROOT
from .triplet import batch_geometry, make_loss, reference_alignment, validate_batch
from .tasks import labels_for


@dataclass
class TrainConfig:
    task: str = "time_label"
    model_path: str = "models/gte-modernbert-base"
    data_dir: str = "data/processed_4090/wcb_time_5768"
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 32
    max_seq_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    margin: float = 0.2
    warmup_ratio: float = 0.1
    eval_every: int = 100
    save_every: int = 100
    log_every: int = 10
    keep_checkpoints: int = 2
    seed: int = 5768
    precision: str = "auto"
    gradient_checkpointing: bool = True
    max_steps: int | None = None
    train_samples_per_label: int | None = None
    val_samples_per_label: int | None = None
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    triplet_mining: str = "semi_hard"
    reference_weight: float = 1.0


def config_from_dict(value):
    """Legacy checkpoints keep their original hard-mining, unanchored objective."""
    return TrainConfig(**{**value, "triplet_mining": value.get("triplet_mining", "hard"),
                          "reference_weight": value.get("reference_weight", 0.)})


@torch.no_grad()
def prepare_references(encoder, splits, output, config, hashes, device, precision):
    """Cache source-encoder outputs before restoring any fine-tuned checkpoint."""
    was_training = encoder.training
    encoder.eval()
    result = {}
    try:
        for split, rows in splits.items():
            fingerprint = {"source": hashes, "max_seq_length": config.max_seq_length,
                           "precision": precision, "sentence_ids": [row["sentence_id"] for row in rows]}
            path = output / f"reference_embeddings_{split}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["fingerprint"] != fingerprint:
                    raise ValueError(f"Source reference inputs changed: {path}")
                result[split] = saved["embeddings"]
                continue
            vectors = []
            for start in range(0, len(rows), config.eval_batch_size):
                sentences = [row["sentence"] for row in rows[start:start + config.eval_batch_size]]
                features = encoder.preprocess(sentences)
                features = {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}
                with autocast(device, precision):
                    embeddings = encoder(features)["sentence_embedding"].float()
                vectors.append(F.normalize(embeddings, p=2, dim=1).cpu())
                completed = min(start + config.eval_batch_size, len(rows))
                if start == 0 or completed == len(rows) or completed % (100 * config.eval_batch_size) == 0:
                    print(json.dumps({"event": "source_reference_cache", "split": split,
                                      "rows_completed": completed, "rows_total": len(rows)}), flush=True)
            embeddings = torch.cat(vectors)
            temporary = path.with_suffix(".pt.tmp")
            with temporary.open("wb") as stream:
                torch.save({"fingerprint": fingerprint, "embeddings": embeddings}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            result[split] = embeddings
    finally:
        encoder.train(was_training)
    return result


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprints(config):
    data_dir, model_dir = resolve_path(config.data_dir), resolve_path(config.model_path)
    if not model_dir.is_dir():
        raise ValueError(f"Download the model first: {model_dir}")
    result = {f"data/{s}.parquet": sha256(data_dir / f"{s}.parquet") for s in ("train", "val")}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.suffix in {".json", ".safetensors", ".bin", ".txt"}:
            if path.name not in {"download_manifest.json", "load_check.json"}:
                result[f"model/{path.relative_to(model_dir)}"] = sha256(path)
    return result


class EventLog:
    def __init__(self, path):
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.session_id = uuid.uuid4().hex

    def write(self, event, console=True, **values):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                  "session_id": self.session_id, "event": event, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.stream.write(line + "\n")
        if console:
            print(line, flush=True)

    def close(self):
        self.stream.close()


def rng_state(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(output, encoder, optimizer, scheduler, progress, config, hashes, precision, device):
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{progress['global_step']:08d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    state = {
        "format_version": 1,
        "config": asdict(config),
        "fingerprints": hashes,
        "precision": precision,
        "progress": dict(progress),
        "model": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(device),
    }
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(output / "latest_checkpoint.json", {"path": str(path.relative_to(output))})
    checkpoints = sorted(directory.glob("step-*.pt"))
    for obsolete in checkpoints[:-config.keep_checkpoints]:
        obsolete.unlink()
    return path


def save_best_model(encoder, output, progress, task="time_label"):
    destination = output / "best_model"
    temporary, backup = output / ".best_model.tmp", output / ".best_model.previous"
    if temporary.exists():
        shutil.rmtree(temporary)
    encoder.save_pretrained(str(temporary), create_model_card=False)
    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    atomic_json(temporary / "wcb_task.json", {
        "task": task, "label_to_id": labels_for(task),
        "max_seq_length": encoder.max_seq_length,
        "global_step": progress["global_step"], "val_loss": progress["best_val_loss"],
        "stage": "supervised_triplet_embedding_finetuning",
        "classification_head_trained": False,
        "triplet_mining": saved_config.get("triplet_mining", "hard"),
        "reference_weight": saved_config.get("reference_weight", 0.),
    })
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def encode_batch(encoder, rows, indices, device):
    selected = [rows[index] for index in indices]
    features = encoder.preprocess([r["sentence"] for r in selected])
    features = {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}
    labels = torch.tensor([r["label"] for r in selected], dtype=torch.long, device=device)
    validate_batch(labels)
    return features, labels


def autocast(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


@torch.no_grad()
def evaluate(encoder, loss_fn, rows, batches, device, precision, *, return_geometry=False,
             reference_embeddings=None, reference_weight=0.):
    was_training = encoder.training
    encoder.eval()
    total, count = 0.0, 0
    geometry = Counter()
    try:
        for indices in batches:
            features, labels = encode_batch(encoder, rows, indices, device)
            with autocast(device, precision):
                embeddings = encoder(features)["sentence_embedding"]
            # Compute cosine distances and the triplet loss in float32.
            triplet_loss = loss_fn.compute_loss_from_embeddings([embeddings.float()], labels)
            reference_loss = (reference_alignment(embeddings, reference_embeddings[indices].to(device))
                              if reference_embeddings is not None else triplet_loss.new_zeros(()))
            loss = triplet_loss + reference_weight * reference_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss is not finite")
            total += loss.item() * len(indices)
            count += len(indices)
            if return_geometry:
                geometry["triplet_loss"] += triplet_loss.item() * len(indices)
                geometry["reference_loss"] += reference_loss.item() * len(indices)
                for key, value in batch_geometry(embeddings, labels, loss_fn.triplet_margin).items():
                    geometry[key] += value * len(indices)
    finally:
        encoder.train(was_training)
    if return_geometry:
        return total / count, {key: value / count for key, value in geometry.items()}
    return total / count


def validate_config(config):
    labels_for(config.task)
    if config.triplet_mining not in {"hard", "semi_hard"}:
        raise ValueError("triplet-mining must be hard or semi_hard")
    if not math.isfinite(config.reference_weight) or config.reference_weight < 0:
        raise ValueError("reference-weight must be finite and nonnegative")
    if config.early_stopping_patience < 0 or not math.isfinite(config.early_stopping_min_delta) or config.early_stopping_min_delta < 0:
        raise ValueError("Early stopping patience/min-delta must be nonnegative")
    for name in ("epochs", "max_seq_length", "eval_every", "save_every", "log_every", "keep_checkpoints"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_steps", "train_samples_per_label", "val_samples_per_label"):
        if getattr(config, name) is not None and getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    if not 0 < config.margin < 2 or not 0 <= config.warmup_ratio < 1:
        raise ValueError("Require 0 < margin < 2 and 0 <= warmup-ratio < 1")
    if config.precision not in {"auto", "bf16", "fp32"}:
        raise ValueError("precision must be auto, bf16, or fp32")


def _train(config, output, *, resume=False, device_name="auto", stop_after_steps=None, cpu_threads=4):
    validate_config(config)
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop-after-steps must be positive")
    if cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    output = Path(output).resolve()
    if not resume and output.exists() and any(p.name != ".training.lock" for p in output.iterdir()):
        raise ValueError(f"Output is not empty; use --resume or a new --output: {output}")
    torch.set_num_threads(cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = ("bf16" if bf16_supported else "fp32") if config.precision == "auto" else config.precision
    if precision == "bf16" and not bf16_supported:
        raise ValueError("bf16 requires a CUDA GPU with bfloat16 support; use --precision fp32")
    data_dir = resolve_path(config.data_dir)
    hashes = fingerprints(config)
    train_rows = load_rows(data_dir / "train.parquet", "train", None, config.seed, config.task)
    val_rows = load_rows(data_dir / "val.parquet", "val", None, config.seed, config.task)
    check_split_overlap(train_rows, val_rows)
    if config.train_samples_per_label is not None:
        train_rows = load_rows(data_dir / "train.parquet", "train", config.train_samples_per_label, config.seed, config.task)
    if config.val_samples_per_label is not None:
        val_rows = load_rows(data_dir / "val.parquet", "val", config.val_samples_per_label, config.seed + 1, config.task)
    train_sampler = BalancedBatchSampler([r["label"] for r in train_rows], config.batch_size, config.seed)
    val_sampler = BalancedBatchSampler([r["label"] for r in val_rows], config.eval_batch_size, config.seed + 2)
    val_batches = val_sampler.batches(0)  # Same anchors/candidates at every evaluation.
    total_steps = min(config.epochs * len(train_sampler), config.max_steps or math.inf)
    total_steps = int(total_steps)
    checkpoint = None
    if resume and (output / "latest_checkpoint.json").exists():
        latest = json.loads((output / "latest_checkpoint.json").read_text())
        checkpoint = torch.load(output / latest["path"], map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1 or asdict(config_from_dict(checkpoint["config"])) != asdict(config):
            raise ValueError("Checkpoint/config mismatch")
        if checkpoint["fingerprints"] != hashes:
            raise ValueError("Model source or train/validation data changed; refusing to resume")
        if checkpoint["precision"] != precision:
            raise ValueError("Resume must use the original resolved precision")
    elif resume and (output / "config.json").exists():
        saved_config = config_from_dict(json.loads((output / "config.json").read_text()))
        if asdict(saved_config) != asdict(config):
            raise ValueError("Configuration changed before the first checkpoint")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    encoder = SentenceTransformer(
        str(resolve_path(config.model_path)), device=str(device), local_files_only=True,
        model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
        config_kwargs={"reference_compile": False},
    )
    if config.max_seq_length > encoder[0].auto_model.config.max_position_embeddings:
        raise ValueError("max-seq-length exceeds model capacity")
    encoder.max_seq_length = config.max_seq_length
    if config.gradient_checkpointing:
        encoder[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    loss_fn = make_loss(encoder, config.margin, config.triplet_mining)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    progress = {"global_step": 0, "epoch": 0, "next_batch": 0,
                "best_val_loss": None, "best_step": None,
                "patience_best_val_loss": None, "bad_evaluations": 0, "early_stopped": False}
    resumed_from_checkpoint = checkpoint is not None
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        atomic_json(output / "config.json", asdict(config))
        atomic_json(output / "data_manifest.json", {
            "fingerprints": hashes,
            "train_rows": len(train_rows), "val_rows": len(val_rows),
            "train_label_counts": dict(Counter(r[config.task] for r in train_rows)),
            "val_label_counts": dict(Counter(r[config.task] for r in val_rows)),
            "batches_per_epoch": len(train_sampler), "total_steps": total_steps,
            "sampling": "Equal label counts; deterministic shuffled cycling of smaller class and final batch.",
            "validation": "Fixed balanced batches; triplet loss plus weighted source-reference alignment.",
            "triplet_mining": config.triplet_mining, "reference_weight": config.reference_weight,
            "test_split_used": False,
        })
    references = (prepare_references(encoder, {"train": train_rows, "val": val_rows}, output,
                                     config, hashes, device, precision)
                  if config.reference_weight else {})
    if checkpoint is not None:
        encoder.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        progress.update(checkpoint["progress"])
        restore_rng(checkpoint["rng"], device)
        del checkpoint
        best = output / "best_model"
        metadata = best / "wcb_task.json"
        exported_step = json.loads(metadata.read_text())["global_step"] if metadata.exists() else None
        if exported_step != progress["best_step"]:
            if progress["best_step"] == progress["global_step"]:
                save_best_model(encoder, output, progress, config.task)
            else:
                raise ValueError("The best_model export is missing or inconsistent with this checkpoint")
    log = EventLog(output / "events.jsonl")
    started = time.monotonic()
    requested_stop = {"signal": None}
    old_handlers = {}

    def request_stop(signum, frame):
        requested_stop["signal"] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)
    last_saved_step = progress["global_step"] if resume else None
    session_steps = 0

    def checkpoint_now():
        nonlocal last_saved_step
        path = save_checkpoint(output, encoder, optimizer, scheduler, progress, config, hashes, precision, device)
        last_saved_step = progress["global_step"]
        log.write("checkpoint", global_step=last_saved_step, path=str(path.relative_to(output)))

    def validation_now():
        value, geometry = evaluate(encoder, loss_fn, val_rows, val_batches, device, precision,
                                   return_geometry=True, reference_embeddings=references.get("val"),
                                   reference_weight=config.reference_weight)
        improved = progress["best_val_loss"] is None or value < progress["best_val_loss"]
        if improved:
            progress["best_val_loss"], progress["best_step"] = value, progress["global_step"]
        reference = progress["patience_best_val_loss"]
        if reference is None or value < reference - config.early_stopping_min_delta:
            progress["patience_best_val_loss"] = value
            progress["bad_evaluations"] = 0
        else:
            progress["bad_evaluations"] += 1
        if config.early_stopping_patience and progress["bad_evaluations"] >= config.early_stopping_patience:
            progress["early_stopped"] = True
        log.write("validation", global_step=progress["global_step"], val_loss=value,
                  improved=improved, best_val_loss=progress["best_val_loss"], best_step=progress["best_step"],
                  bad_evaluations=progress["bad_evaluations"], early_stopped=progress["early_stopped"],
                  **geometry)
        return improved

    try:
        log.write("resume" if resume else "start", global_step=progress["global_step"],
                  total_steps=total_steps, device=str(device), precision=precision,
                  triplet_mining=config.triplet_mining, reference_weight=config.reference_weight,
                  python=platform.python_version(), torch=str(torch.__version__),
                  transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__)
        if not resumed_from_checkpoint:
            validation_now()  # The unmodified encoder is also eligible to be best.
            checkpoint_now()
            save_best_model(encoder, output, progress, config.task)
        encoder.train()
        while progress["global_step"] < total_steps and not requested_stop["signal"] and not progress["early_stopped"]:
            epoch = progress["epoch"]
            batches = train_sampler.batches(epoch)
            for batch_index in range(progress["next_batch"], len(batches)):
                if requested_stop["signal"]:
                    break
                indices = batches[batch_index]
                features, labels = encode_batch(encoder, train_rows, indices, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, precision):
                    embeddings = encoder(features)["sentence_embedding"]
                triplet_loss = loss_fn.compute_loss_from_embeddings([embeddings.float()], labels)
                reference_loss = (reference_alignment(embeddings, references["train"][indices].to(device))
                                  if references else triplet_loss.new_zeros(()))
                loss = triplet_loss + config.reference_weight * reference_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss is not finite")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0, error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                progress["global_step"] += 1
                session_steps += 1
                epoch_end = batch_index + 1 == len(batches)
                progress["epoch"] = epoch + 1 if epoch_end else epoch
                progress["next_batch"] = 0 if epoch_end else batch_index + 1
                step = progress["global_step"]
                log.write("train", console=step % config.log_every == 0 or step == 1,
                          global_step=step, epoch=epoch + 1, batch_in_epoch=batch_index + 1,
                          train_loss=loss.item(), learning_rate=learning_rate, grad_norm=grad_norm.item(),
                          triplet_loss=triplet_loss.item(), reference_loss=reference_loss.item(),
                          batch_label_counts={str(k): v for k, v in zip(
                              *[part.tolist() for part in torch.unique(labels, return_counts=True)])},
                          session_seconds=time.monotonic() - started)
                improved = False
                if step % config.eval_every == 0 or epoch_end or step == total_steps:
                    improved = validation_now()
                if step % config.save_every == 0 or improved:
                    checkpoint_now()
                if improved:
                    save_best_model(encoder, output, progress, config.task)
                if progress["early_stopped"] or step >= total_steps or (stop_after_steps is not None and session_steps >= stop_after_steps):
                    break
            if stop_after_steps is not None and session_steps >= stop_after_steps:
                break
        if last_saved_step != progress["global_step"]:
            checkpoint_now()
        result = {
            "status": "early_stopped" if progress["early_stopped"] else ("completed" if progress["global_step"] >= total_steps else "stopped"),
            **progress, "total_steps": total_steps, "session_steps": session_steps,
            "stop_signal": requested_stop["signal"], "session_seconds": time.monotonic() - started,
            "best_model": str(output / "best_model"),
            "latest_checkpoint": json.loads((output / "latest_checkpoint.json").read_text())["path"],
            "classification_head_trained": False, "test_split_used": False,
            "triplet_mining": config.triplet_mining, "reference_weight": config.reference_weight,
        }
        atomic_json(output / "summary.json", result)
        log.write(result["status"], **{k: v for k, v in result.items() if k != "status"})
        return result
    except BaseException as exc:
        log.write("error", global_step=progress["global_step"], error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        log.close()


def train(config, output, **runtime):
    """Keep two processes from writing to the same run (Linux/Vast.ai)."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another training process is using {output}") from exc
        try:
            return _train(config, output, **runtime)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/wcb-time")
    parser.add_argument("--resume", action="store_true", help="Restore config and latest checkpoint from --output")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stop-after-steps", type=int, help="Save and stop after this many additional steps; training schedule is unchanged")
    defaults = asdict(TrainConfig())
    for name, default in defaults.items():
        kwargs = {"default": argparse.SUPPRESS}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = int if default is None else type(default)
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    args = vars(parser.parse_args(argv))
    output = resolve_path(args.pop("output"))
    runtime = {key: args.pop(key) for key in ("resume", "device", "cpu_threads", "stop_after_steps")}
    if runtime["resume"]:
        saved = asdict(config_from_dict(json.loads((output / "config.json").read_text())))
        for key, value in args.items():
            if value != saved[key]:
                parser.error(f"Cannot change --{key.replace('_', '-')} during resume")
        config = TrainConfig(**saved)
    else:
        config = TrainConfig(**args)
    return train(config, output, resume=runtime["resume"], device_name=runtime["device"],
                 cpu_threads=runtime["cpu_threads"], stop_after_steps=runtime["stop_after_steps"])


if __name__ == "__main__":
    main()
