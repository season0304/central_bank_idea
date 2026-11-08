"""Train all three dense classifiers locally from downloaded triplet encoders."""
import argparse
import fcntl
import gc
import json

import torch

from .classifier import ClassifierConfig, evaluate_test, train
from .tasks import LABELS_BY_TASK, labels_for
from .train import atomic_json, resolve_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/wcb-4090-all", help="Downloaded run folder containing each task's triplet/best_model")
    parser.add_argument("--data-root", default="data/processed_4090")
    parser.add_argument("--data-seed", default="5768")
    parser.add_argument("--tasks", nargs="+", choices=list(LABELS_BY_TASK), default=list(LABELS_BY_TASK))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--feature-scaling", choices=["standard", "none"],
                        help="Default: standard for new runs; saved setting when resuming")
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5768)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Each task must occur once")
    root = resolve_path(args.run)
    if not root.is_dir():
        parser.error(f"Download the triplet run directory first: {root}")
    configs = {}
    for task in args.tasks:
        encoder = root / task / "triplet" / "best_model"
        metadata = encoder / "wcb_task.json"
        if not metadata.exists():
            parser.error(f"Missing trained encoder: {metadata}")
        info = json.loads(metadata.read_text())
        if info.get("task") != task or info.get("label_to_id") != labels_for(task):
            parser.error(f"Encoder task/labels do not match: {metadata}")
        triplet_summary = encoder.parent / "summary.json"
        if triplet_summary.exists() and json.loads(triplet_summary.read_text())["status"] not in {"completed", "early_stopped"}:
            parser.error(f"Triplet run has not finished: {triplet_summary}")
        data_dir = resolve_path(args.data_root) / f"wcb_{task.removesuffix('_label')}_{args.data_seed}"
        for split in ("train", "val") + (() if args.skip_test else ("test",)):
            if not (data_dir / f"{split}.parquet").exists():
                parser.error(f"Missing {data_dir / (split + '.parquet')}")
        config = ClassifierConfig(task=task, encoder_path=str(encoder), data_dir=str(data_dir),
            epochs=args.epochs, early_stopping_patience=args.patience, selection_metric=args.selection_metric,
            batch_size=args.batch_size, embedding_batch_size=args.embedding_batch_size,
            seed=args.seed, precision=args.precision, max_seq_length=info.get("max_seq_length", 256))
        saved_config = root / task / "classifier" / "config.json"
        if args.feature_scaling is not None:
            config.feature_scaling = args.feature_scaling
        elif args.resume and saved_config.exists():
            config.feature_scaling = json.loads(saved_config.read_text()).get("feature_scaling", "none")
        # Older exports omit max_seq_length; recover it from the encoder's own config.
        encoder_config = encoder / 'sentence_bert_config.json'
        if encoder_config.exists():
            config.max_seq_length = json.loads(encoder_config.read_text()).get('max_seq_length', config.max_seq_length)
        configs[task] = config
    lock = (root / ".local_classifiers.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        parser.error("Another local classifier pipeline is using this run")
    results = {}
    try:
        # This also permits prediction when only the three best_model folders were downloaded.
        if not (root / "pipeline_config.json").exists():
            atomic_json(root / "pipeline_config.json", {"tasks": args.tasks})
        for task, config in configs.items():
            output = root / task / "classifier"
            print(f"Local classifier: {task}; device={args.device}; encoder frozen", flush=True)
            result = train(config, output, device_name=args.device,
                           resume=args.resume and (output / "config.json").exists(), cpu_threads=args.cpu_threads)
            results[task] = result
            if result["status"] not in {"completed", "early_stopped"}:
                atomic_json(root / "local_classifiers_summary.json", {"status": "stopped", "tasks": results})
                return 0
            if not args.skip_test:
                results[task]["test"] = evaluate_test(output, args.device)
            atomic_json(root / "local_classifiers_summary.json", {"status": "running", "tasks": results})
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        atomic_json(root / "local_classifiers_summary.json", {"status": "completed", "tasks": results})
        print('Local classifiers completed:', root / 'local_classifiers_summary.json', flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    main()
