"""Exercise sampling and exact restart behavior with a tiny, local encoder."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer, models

from wcb_triplet.batching import BalancedBatchSampler
from wcb_triplet.train import main


def create_tiny_encoder(path):
    torch.manual_seed(123)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "bank", "expect", "inflation",
             "next", "year", "was", "last", "month", "rise", "fall"]
    words += [str(i) for i in range(100)]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=64)
    source = path.parent / "tiny_backbone"
    source.mkdir()
    fast.save_pretrained(source)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
    BertModel(config).save_pretrained(source)
    encoder = SentenceTransformer(modules=[models.Transformer(str(source)), models.Pooling(16)], device="cpu")
    encoder.save_pretrained(str(path), create_model_card=False)


def create_data(path):
    path.mkdir()
    for split, per_label, offset in (("train", 6, 0), ("val", 4, 20)):
        rows = []
        for label in (0, 1):
            for i in range(per_label):
                ending = "expect inflation next year" if label else "inflation was last month"
                rows.append({"sentence_id": f"{split}:{label}:{i}",
                             "sentence": f"bank {offset+i} {ending} {'rise' if i % 2 else 'fall'}",
                             "label": label,
                             "time_label": "forward looking" if label else "not forward looking",
                             "source_split": split})
        pq.write_table(pa.Table.from_pylist(rows), path / f"{split}.parquet")
    # There is intentionally no test.parquet: model selection must not need it.


def latest_state(output):
    pointer = json.loads((output / "latest_checkpoint.json").read_text())
    return torch.load(output / pointer["path"], weights_only=True, map_location="cpu")


class BatchTests(unittest.TestCase):
    def test_balanced_unique_reproducible_and_majority_coverage(self):
        labels = [0] * 11 + [1] * 5
        sampler = BalancedBatchSampler(labels, batch_size=8, seed=42)
        batches = sampler.batches(0)
        self.assertEqual(batches, sampler.batches(0))
        self.assertNotEqual(batches, sampler.batches(1))
        seen = set()
        for batch in batches:
            self.assertEqual(len(batch), len(set(batch)))
            self.assertEqual(Counter(labels[i] for i in batch), {0: 4, 1: 4})
            seen.update(batch)
        self.assertEqual(seen, set(range(len(labels))))

    def test_impossible_batches_rejected(self):
        for labels, batch_size in (([0] * 8, 4), ([0] * 4 + [1], 4), ([0] * 4 + [1] * 4, 5)):
            with self.assertRaises(ValueError):
                BalancedBatchSampler(labels, batch_size, seed=1)


class TrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA bfloat16 unavailable")
    def test_cuda_bf16_training_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, data, output = root / "model", root / "data", root / "run"
            create_tiny_encoder(model)
            create_data(data)
            options = ["--model-path", str(model), "--data-dir", str(data), "--device", "cuda",
                       "--precision", "bf16", "--batch-size", "4", "--eval-batch-size", "4",
                       "--max-seq-length", "32", "--epochs", "1", "--max-steps", "2",
                       "--eval-every", "1", "--save-every", "1", "--cpu-threads", "1"]
            with redirect_stdout(io.StringIO()):
                main(["--output", str(output), "--stop-after-steps", "1", *options])
                result = main(["--output", str(output), "--resume", "--device", "cuda", "--cpu-threads", "1"])
            state = latest_state(output)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(state["precision"], "bf16")
            self.assertIsNotNone(state["rng"]["cuda"])
            self.assertEqual(state["progress"]["global_step"], 2)
            self.assertTrue(all(torch.isfinite(t).all() for t in state["model"].values()))

    def test_restart_matches_uninterrupted_training_and_exports_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, data = root / "model", root / "data"
            create_tiny_encoder(model)
            create_data(data)
            initial_weights = SentenceTransformer(str(model), device="cpu").state_dict()
            full, resumed = root / "full", root / "resumed"
            options = ["--model-path", str(model), "--data-dir", str(data), "--device", "cpu",
                       "--cpu-threads", "1", "--precision", "fp32", "--batch-size", "4",
                       "--eval-batch-size", "4", "--max-seq-length", "32", "--epochs", "2",
                       "--max-steps", "5", "--eval-every", "2", "--save-every", "2",
                       "--learning-rate", "0.001", "--warmup-ratio", "0.4"]
            with redirect_stdout(io.StringIO()):
                finished = main(["--output", str(full), *options])
                stopped = main(["--output", str(resumed), "--stop-after-steps", "2", *options])
                self.assertEqual(stopped["status"], "stopped")
                self.assertEqual(stopped["global_step"], 2)
                saved_config = (resumed / "config.json").read_bytes()
                restarted = main(["--output", str(resumed), "--resume", "--device", "cpu", "--cpu-threads", "1"])
            self.assertEqual(restarted["status"], "completed")
            self.assertEqual(restarted["global_step"], 5)
            self.assertEqual(saved_config, (resumed / "config.json").read_bytes())
            a, b = latest_state(full), latest_state(resumed)
            self.assertEqual(a["progress"], b["progress"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertTrue(any(not torch.equal(initial_weights[key], a["model"][key]) for key in a["model"]))
            for parameter, state in a["optimizer"]["state"].items():
                for key, value in state.items():
                    other = b["optimizer"]["state"][parameter][key]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, other), key)
                    else:
                        self.assertEqual(value, other)
            events = [json.loads(line) for line in (resumed / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["global_step"] for e in events if e["event"] == "train"], [1, 2, 3, 4, 5])
            self.assertTrue(all(e["batch_label_counts"] == {"0": 2, "1": 2} for e in events if e["event"] == "train"))
            evaluations = [e for e in events if e["event"] == "validation"]
            self.assertEqual(finished["best_val_loss"], min(e["val_loss"] for e in evaluations))
            self.assertLessEqual(len(list((resumed / "checkpoints").glob("*.pt"))), 2)
            best = SentenceTransformer(str(resumed / "best_model"), device="cpu", local_files_only=True)
            self.assertEqual(best.encode(["bank inflation next year"]).shape, (1, 16))
            with self.assertRaises(ValueError):
                main(["--output", str(resumed), *options])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--output", str(resumed), "--resume", "--batch-size", "8"])
            with (resumed / ".training.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "Another training process"):
                    main(["--output", str(resumed), "--resume", "--device", "cpu"])
                fcntl.flock(lock, fcntl.LOCK_UN)
            changed = pq.read_table(data / "val.parquet").to_pylist()
            changed[0]["sentence"] += " changed"
            pq.write_table(pa.Table.from_pylist(changed), data / "val.parquet")
            with self.assertRaisesRegex(ValueError, "data changed"):
                main(["--output", str(resumed), "--resume", "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
