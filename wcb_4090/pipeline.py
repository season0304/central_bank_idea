"""Train all three WCB tasks sequentially, including their dense classifiers."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys

from .classifier import ClassifierConfig
from .prepare import PROJECT_ROOT
from .tasks import LABELS_BY_TASK
from .train import TrainConfig, atomic_json, resolve_path


def portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/wcb-4090-all")
    parser.add_argument("--model-path", default="models/gte-modernbert-base")
    parser.add_argument("--data-root", default="data/processed_4090")
    parser.add_argument("--data-seed", default="5768")
    parser.add_argument("--tasks", nargs="+", choices=list(LABELS_BY_TASK), default=list(LABELS_BY_TASK))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--triplet-epochs", type=int, default=3)
    parser.add_argument("--triplet-eval-every", type=int, default=200)
    parser.add_argument("--triplet-patience", type=int, default=5)
    parser.add_argument("--triplet-min-delta", type=float, default=0.001)
    parser.add_argument("--triplet-mining", choices=["hard", "semi_hard"],
                        help="Default: semi_hard for new runs; saved setting on resume")
    parser.add_argument("--reference-weight", type=float,
                        help="Source embedding preservation weight; default 1 for new runs")
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-patience", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5768)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--triplet-only", action="store_true", help="Train only the three encoders; train heads locally afterwards")
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Each task must occur once")
    root = resolve_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_config.json"
    saved_config = json.loads(state_path.read_text()) if state_path.exists() else None
    if saved_config is not None:
        saved_config.setdefault("triplet_mining", "hard")
        saved_config.setdefault("reference_weight", 0.)
    if args.triplet_mining is None:
        args.triplet_mining = saved_config["triplet_mining"] if saved_config is not None else "semi_hard"
    if args.reference_weight is None:
        args.reference_weight = saved_config["reference_weight"] if saved_config is not None else 1.
    config = vars(args).copy()
    config.pop("resume")
    config.pop("device")  # Device may change at a restart; resolved precision may not.
    config.pop("cpu_threads")
    if state_path.exists():
        if not args.resume:
            parser.error("Output already exists; run with --resume")
        if saved_config != config:
            parser.error("Pipeline settings changed; use the original arguments or a new output")
    for task in args.tasks:
        directory = resolve_path(args.data_root) / f"wcb_{task.removesuffix('_label')}_{args.data_seed}"
        for split in ("train", "val") + (() if args.skip_test or args.triplet_only else ("test",)):
            if not (directory / f"{split}.parquet").exists():
                parser.error(f"Missing data: {directory / (split + '.parquet')}; prepare all tasks first")
    lock = (root / ".pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        parser.error("Another pipeline is using this output")
    atomic_json(state_path, config)
    child = {"process": None, "signal": None}
    def stop(signum, frame):
        child["signal"] = signum
        if child["process"] is not None:
            child["process"].send_signal(signum)
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    stages = []
    def run_stage(module, output, options):
        if child["signal"]:
            return False
        command = [sys.executable, "-m", module, "--output", str(output),
                   "--device", args.device, "--cpu-threads", str(args.cpu_threads)]
        if args.resume and (output / "config.json").exists():
            command.append("--resume")
        else:
            for name, value in options.items():
                if isinstance(value, bool):
                    command.append("--" + ("" if value else "no-") + name.replace("_", "-"))
                elif value is not None:
                    command.extend(["--" + name.replace("_", "-"), str(value)])
        if module.endswith("classifier") and not args.skip_test:
            command.append("--evaluate-test")
        print("Running:", " ".join(command), flush=True)
        child["process"] = subprocess.Popen(command, cwd=PROJECT_ROOT, start_new_session=True)
        code = child["process"].wait()
        child["process"] = None
        if code:
            raise RuntimeError(f"{module} failed with exit code {code}; correct the cause and use --resume")
        result = json.loads((output / "summary.json").read_text())
        stages.append({"module": module, "output": portable(output), "status": result["status"]})
        atomic_json(root / "pipeline_summary.json", {"status": "running", "stages": stages})
        return not child["signal"] and result["status"] in {"completed", "early_stopped"}
    try:
        for task in args.tasks:
            folder = root / task
            data = portable(resolve_path(args.data_root) / f"wcb_{task.removesuffix('_label')}_{args.data_seed}")
            triplet = TrainConfig(task=task, model_path=args.model_path, data_dir=data,
                                  epochs=args.triplet_epochs, batch_size=args.batch_size,
                                  eval_batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                                  eval_every=args.triplet_eval_every, early_stopping_patience=args.triplet_patience,
                                  triplet_mining=args.triplet_mining, reference_weight=args.reference_weight,
                                  early_stopping_min_delta=args.triplet_min_delta, precision=args.precision, seed=args.seed)
            if not run_stage("wcb_4090.train", folder / "triplet", asdict(triplet)):
                atomic_json(root / "pipeline_summary.json", {"status": "stopped", "stages": stages})
                return 0
            if args.triplet_only:
                continue
            head = ClassifierConfig(task=task, encoder_path=portable(folder / "triplet" / "best_model"),
                                    data_dir=data, epochs=args.classifier_epochs,
                                    early_stopping_patience=args.classifier_patience,
                                    selection_metric=args.selection_metric, max_seq_length=args.max_seq_length,
                                    embedding_batch_size=args.embedding_batch_size,
                                    precision=args.precision, seed=args.seed)
            if not run_stage("wcb_4090.classifier", folder / "classifier", asdict(head)):
                atomic_json(root / "pipeline_summary.json", {"status": "stopped", "stages": stages})
                return 0
        summary = {"status": "completed", "stages": stages,
                   "test_results": {task: json.loads((root / task / "classifier" / "test_metrics.json").read_text())
                                    for task in args.tasks} if not args.skip_test and not args.triplet_only else {},
                   "triplet_only": args.triplet_only}
        atomic_json(root / "pipeline_summary.json", summary)
        print("All tasks completed:", root / "pipeline_summary.json", flush=True)
        return 0
    except BaseException as exc:
        atomic_json(root / "pipeline_summary.json", {"status": "error", "stages": stages,
                                                     "error": str(exc)})
        raise
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
