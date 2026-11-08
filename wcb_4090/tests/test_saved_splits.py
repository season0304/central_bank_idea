"""Test leakage cohorts, including duplicate text under different source IDs."""
import unittest

from wcb_4090.evaluate_saved_splits import overlap_cohorts


class SavedSplitsTests(unittest.TestCase):
    def test_excludes_training_and_validation_by_id_or_text(self):
        train = [{'sentence_id': 'train:1', 'sentence': 'Training sentence'}]
        val = [{'sentence_id': 'val:1', 'sentence': 'Validation sentence'}]
        rows = [{'sentence_id': 'train:1', 'sentence': 'Training sentence'},
                {'sentence_id': 'different-id', 'sentence': ' Training  sentence '},
                {'sentence_id': 'val:1', 'sentence': 'Validation sentence'},
                {'sentence_id': 'another-id', 'sentence': 'Validation  sentence'},
                {'sentence_id': 'test:1', 'sentence': 'Unseen sentence'}]
        result = overlap_cohorts(rows, train, val)
        self.assertEqual(result['seen_in_training'], [0, 1])
        self.assertEqual(result['seen_in_validation_only'], [2, 3])
        self.assertEqual(result['unseen_in_training_and_validation'], [4])
        self.assertEqual(result['all'], list(range(5)))


if __name__ == '__main__':
    unittest.main()
