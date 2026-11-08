"""Verify the incompatible published/project label order before any inference."""
import unittest

import torch

from wcb_4090.benchmark_stance import paired_outcomes, remap_logits
from wcb_4090.classifier import metrics


class BenchmarkTests(unittest.TestCase):
    def test_published_label_order_is_remapped(self):
        logits = torch.eye(4) * 10
        labels = torch.tensor([2, 1, 0, 3])  # neutral, hawkish, dovish, irrelevant
        converted = remap_logits(logits)
        self.assertEqual(converted.argmax(1).tolist(), labels.tolist())
        self.assertEqual(metrics(converted, labels, 4)["accuracy"], 1.)
        self.assertLess(metrics(logits, labels, 4)["accuracy"], 1.)

    def test_paired_outcomes_keep_correct_denominator(self):
        result = paired_outcomes(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 1, 0]))
        self.assertEqual(result, {"both_correct": 1, "local_only_correct": 1,
                                  "published_only_correct": 1, "both_wrong": 1})


if __name__ == "__main__":
    unittest.main()
