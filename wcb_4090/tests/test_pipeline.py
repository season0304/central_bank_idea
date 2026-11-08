"""CPU-only integration tests with a tiny encoder; never train the production GTE."""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer, models

from wcb_4090.batching import BalancedBatchSampler
from wcb_4090.classifier import (ClassifierConfig, config_from_dict, evaluate_test, metrics,
                                 train as train_classifier, transform_features)
from wcb_4090.pipeline import main as pipeline
from wcb_4090.local_classifiers import main as local_classifiers
from wcb_4090.predict import predict
from wcb_4090.tasks import LABELS_BY_TASK
from wcb_4090.train import main as triplet
from wcb_4090.triplet import batch_geometry, make_loss, reference_alignment


def tiny_encoder(path):
    torch.manual_seed(123)
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'bank', 'future', 'past', 'certain', 'uncertain', 'policy']
    words += [str(i) for i in range(150)]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='[CLS] $A [SEP]',
        special_tokens=[('[CLS]', vocab['[CLS]']), ('[SEP]', vocab['[SEP]'])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]',
        cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', model_max_length=64)
    source = path.parent / 'backbone'
    source.mkdir()
    fast.save_pretrained(source)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)).save_pretrained(source)
    SentenceTransformer(modules=[models.Transformer(str(source)), models.Pooling(16)], device='cpu').save_pretrained(str(path), create_model_card=False)


def data(root, task, *, with_test=True):
    folder = root / f"wcb_{task.removesuffix('_label')}_5768"
    folder.mkdir(parents=True)
    for split, offset in [('train', 0), ('val', 50)] + ([('test', 100)] if with_test else []):
        rows = []
        for name, label in LABELS_BY_TASK[task].items():
            for i in range(4):
                rows.append({'sentence_id': f'{split}:{label}:{i}',
                    'sentence': f'bank {offset + label * 4 + i} policy {name}',
                    'label': label, task: name, 'source_split': split})
        pq.write_table(pa.Table.from_pylist(rows), folder / f'{split}.parquet')
    return folder


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_four_class_batches_and_metrics(self):
        labels = [0] * 8 + [1] * 5 + [2] * 4 + [3] * 3
        sampler = BalancedBatchSampler(labels, 8, 42)
        seen = set()
        for batch in sampler.batches(0):
            self.assertEqual(len(batch), len(set(batch)))
            self.assertEqual(Counter(labels[i] for i in batch), {0: 2, 1: 2, 2: 2, 3: 2})
            seen.update(batch)
        self.assertEqual(seen, set(range(len(labels))))
        value = metrics(torch.tensor([[3., 0.], [0., 3.], [0., 3.]]), torch.tensor([0, 1, 0]), 2)
        self.assertEqual(value['confusion_matrix'], [[1, 1], [0, 1]])
        self.assertAlmostEqual(value['macro_f1'], 2 / 3, places=6)

    def test_triplet_early_stop_survives_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            source = data(root / 'data', 'stance_label', with_test=False)
            output = root / 'run'
            args = ['--output', str(output), '--model-path', str(model), '--data-dir', str(source),
                    '--task', 'stance_label', '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
                    '--batch-size', '8', '--eval-batch-size', '8', '--max-seq-length', '32', '--epochs', '3',
                    '--eval-every', '1', '--early-stopping-patience', '1', '--early-stopping-min-delta', '10']
            with redirect_stdout(io.StringIO()):
                result = triplet(args)
                resumed = triplet(['--output', str(output), '--resume', '--device', 'cpu', '--cpu-threads', '1'])
            self.assertEqual(result['status'], 'early_stopped')
            self.assertEqual(result['global_step'], 1)
            self.assertEqual(resumed['global_step'], 1)
            metadata = json.loads((output / 'best_model/wcb_task.json').read_text())
            self.assertEqual(metadata['task'], 'stance_label')
            self.assertEqual(len(metadata['label_to_id']), 4)

    def test_reference_preservation_penalizes_constant_vectors_and_has_gradient(self):
        reference = torch.eye(4)
        labels = torch.tensor([0, 0, 1, 1])
        collapsed = torch.ones(4, 4, requires_grad=True)
        aligned = reference.clone().requires_grad_(True)
        self.assertEqual(reference_alignment(aligned, reference).item(), 0.)
        self.assertGreater(reference_alignment(collapsed, reference).item(), .4)
        loss_fn = make_loss(torch.nn.Identity(), .2, 'semi_hard')
        collapsed_loss = loss_fn.compute_loss_from_embeddings([collapsed], labels) + reference_alignment(collapsed, reference)
        aligned_loss = loss_fn.compute_loss_from_embeddings([aligned], labels) + reference_alignment(aligned, reference)
        self.assertGreater(collapsed_loss.item(), aligned_loss.item())
        collapsed_loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())
        self.assertGreater(collapsed.grad.norm().item(), 0.)

    def test_reference_triplets_restart_exactly_and_regenerate_original_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            source = data(root / 'data', 'time_label', with_test=False)
            options = ['--model-path', str(model), '--data-dir', str(source), '--device', 'cpu',
                       '--cpu-threads', '1', '--precision', 'fp32', '--batch-size', '4',
                       '--eval-batch-size', '4', '--max-seq-length', '32', '--epochs', '3',
                       '--max-steps', '5', '--eval-every', '2', '--save-every', '2',
                       '--learning-rate', '.001', '--warmup-ratio', '.4']
            with redirect_stdout(io.StringIO()):
                full = triplet(['--output', str(root / 'full'), *options])
                triplet(['--output', str(root / 'resume'), '--stop-after-steps', '2', *options])
                # A missing reference cache must be rebuilt from the source, never the resumed student.
                for split in ('train', 'val'):
                    (root / 'resume' / f'reference_embeddings_{split}.pt').unlink()
                resumed = triplet(['--output', str(root / 'resume'), '--resume', '--device', 'cpu', '--cpu-threads', '1'])
            self.assertEqual(full['status'], 'completed')
            self.assertEqual(full['triplet_mining'], 'semi_hard')
            self.assertEqual(full['reference_weight'], 1.)
            self.assertEqual(resumed['global_step'], 5)
            states = []
            for name in ('full', 'resume'):
                directory = root / name
                pointer = json.loads((directory / 'latest_checkpoint.json').read_text())
                states.append(torch.load(directory / pointer['path'], weights_only=True))
            self.assertEqual(states[0]['progress'], states[1]['progress'])
            for key in states[0]['model']:
                self.assertTrue(torch.equal(states[0]['model'][key], states[1]['model'][key]), key)
            for split in ('train', 'val'):
                a = torch.load(root / 'full' / f'reference_embeddings_{split}.pt', weights_only=True)
                b = torch.load(root / 'resume' / f'reference_embeddings_{split}.pt', weights_only=True)
                self.assertTrue(torch.equal(a['embeddings'], b['embeddings']))
            events = [json.loads(line) for line in (root / 'full/events.jsonl').read_text().splitlines()]
            self.assertTrue(all('reference_loss' in event for event in events if event['event'] == 'validation'))

    def test_legacy_triplet_checkpoint_keeps_hard_mining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            source = data(root / 'data', 'time_label', with_test=False)
            output = root / 'run'
            options = ['--output', str(output), '--model-path', str(model), '--data-dir', str(source),
                       '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
                       '--batch-size', '4', '--eval-batch-size', '4', '--max-seq-length', '32',
                       '--max-steps', '2', '--triplet-mining', 'hard', '--reference-weight', '0']
            with redirect_stdout(io.StringIO()):
                triplet([*options, '--stop-after-steps', '1'])
            config = json.loads((output / 'config.json').read_text())
            pointer = json.loads((output / 'latest_checkpoint.json').read_text())
            checkpoint = torch.load(output / pointer['path'], weights_only=True)
            for name in ('triplet_mining', 'reference_weight'):
                config.pop(name)
                checkpoint['config'].pop(name)
            (output / 'config.json').write_text(json.dumps(config))
            torch.save(checkpoint, output / pointer['path'])
            with redirect_stdout(io.StringIO()):
                result = triplet(['--output', str(output), '--resume', '--device', 'cpu', '--cpu-threads', '1'])
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['triplet_mining'], 'hard')
            self.assertEqual(result['reference_weight'], 0.)

    def test_legacy_pipeline_resume_keeps_original_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            data(root / 'data', 'time_label', with_test=False)
            output = root / 'run'
            args = ['--output', str(output), '--model-path', str(model), '--data-root', str(root / 'data'),
                    '--tasks', 'time_label', '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
                    '--batch-size', '4', '--max-seq-length', '32', '--triplet-epochs', '1', '--triplet-only']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline([*args, '--triplet-mining', 'hard', '--reference-weight', '0']), 0)
            triplet_dir = output / 'time_label/triplet'
            for file in (output / 'pipeline_config.json', triplet_dir / 'config.json'):
                saved = json.loads(file.read_text())
                for key in ('triplet_mining', 'reference_weight'):
                    saved.pop(key)
                file.write_text(json.dumps(saved))
            pointer = json.loads((triplet_dir / 'latest_checkpoint.json').read_text())
            checkpoint = torch.load(triplet_dir / pointer['path'], weights_only=True)
            for key in ('triplet_mining', 'reference_weight'):
                checkpoint['config'].pop(key)
            torch.save(checkpoint, triplet_dir / pointer['path'])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline([*args, '--resume']), 0)
            saved = json.loads((output / 'pipeline_config.json').read_text())
            self.assertEqual(saved['triplet_mining'], 'hard')
            self.assertEqual(saved['reference_weight'], 0.)

    def test_classifier_resume_exact_and_test_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            source = data(root / 'data', 'certain_label', with_test=False)
            config = ClassifierConfig(task='certain_label', encoder_path=str(model), data_dir=str(source),
                epochs=3, batch_size=4, embedding_batch_size=4, max_seq_length=32,
                precision='fp32', early_stopping_patience=10)
            with redirect_stdout(io.StringIO()):
                full = train_classifier(ClassifierConfig(**asdict(config)), root / 'full', device_name='cpu', cpu_threads=1)
                stopped = train_classifier(ClassifierConfig(**asdict(config)), root / 'resume', device_name='cpu', stop_after_steps=1, cpu_threads=1)
                resumed = train_classifier(ClassifierConfig(**asdict(config)), root / 'resume', device_name='cpu', resume=True, cpu_threads=1)
            self.assertEqual(stopped['status'], 'stopped')
            self.assertEqual(resumed['status'], 'completed')
            self.assertFalse(full['test_split_used'])
            a = torch.load(root / 'full/checkpoint.pt', weights_only=True)
            b = torch.load(root / 'resume/checkpoint.pt', weights_only=True)
            self.assertEqual(a['progress'], b['progress'])
            for key in a['head']:
                self.assertTrue(torch.equal(a['head'][key], b['head'][key]))
            for key in ('mean', 'scale'):
                self.assertTrue(torch.equal(a['feature_transform'][key], b['feature_transform'][key]))
            self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
            # Test data is absent throughout model selection; add it only afterwards.
            rows = pq.read_table(source / 'val.parquet').to_pylist()
            for row in rows:
                row['source_split'] = 'test'; row['sentence_id'] = 'test:' + row['sentence_id']; row['sentence'] += ' 149'
            pq.write_table(pa.Table.from_pylist(rows), source / 'test.parquet')
            value = evaluate_test(root / 'full', 'cpu')
            self.assertEqual(value['test_rows'], 8)
            self.assertEqual(evaluate_test(root / 'full', 'cpu'), value)
            rows[0]['sentence'] += ' changed'
            pq.write_table(pa.Table.from_pylist(rows), source / 'test.parquet')
            with self.assertRaisesRegex(ValueError, 'Test result inputs changed'):
                evaluate_test(root / 'full', 'cpu')

    def test_standardization_learns_nearly_identical_embeddings_and_uses_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = data(root / 'data', 'time_label', with_test=False)
            cached = {}
            for split in ('train', 'val'):
                labels = torch.tensor([row['label'] for row in pq.read_table(source / f'{split}.parquet').to_pylist()])
                x = torch.ones(len(labels), 3)
                x[:, 1] = .001 * (labels * 2 - 1) + (.0002 if split == 'val' else 0)
                x[:, 2] = 0
                x = torch.nn.functional.normalize(x, dim=1)
                cached[split] = {'embeddings': x, 'labels': labels, 'fingerprint': {'split': split}}
            config = ClassifierConfig(task='time_label', encoder_path='unused', data_dir=str(source),
                precision='fp32', epochs=15, batch_size=4, learning_rate=.05,
                weight_decay=0, early_stopping_patience=0)
            with patch('wcb_4090.classifier.make_embeddings', return_value=cached), redirect_stdout(io.StringIO()):
                result = train_classifier(config, root / 'run', device_name='cpu', cpu_threads=1)
            self.assertEqual(result['epoch'], 15)
            self.assertFalse(result['early_stopped'])
            artifact = torch.load(root / 'run/best_classifier.pt', weights_only=True)
            transform = artifact['feature_transform']
            self.assertTrue(torch.equal(transform['mean'], cached['train']['embeddings'].mean(0)))
            self.assertFalse(torch.equal(transform['mean'], cached['val']['embeddings'].mean(0)))
            head = torch.nn.Linear(3, 2)
            head.load_state_dict(artifact['head'])
            value = metrics(head(transform_features(cached['val']['embeddings'], transform)), cached['val']['labels'], 2)
            self.assertEqual(value['accuracy'], 1.)
            self.assertTrue(torch.isfinite(transform_features(cached['val']['embeddings'], transform)).all())

    def test_legacy_classifier_config_and_artifact_keep_unscaled_features(self):
        config = config_from_dict({'task': 'time_label'})
        self.assertEqual(config.feature_scaling, 'none')
        vectors = torch.randn(8, 3)
        self.assertTrue(torch.equal(transform_features(vectors), vectors))

    def test_legacy_classifier_checkpoint_resumes_without_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            source = data(root / 'data', 'time_label', with_test=False)
            config = ClassifierConfig(task='time_label', encoder_path=str(model), data_dir=str(source),
                epochs=1, batch_size=4, embedding_batch_size=4, max_seq_length=32,
                precision='fp32', early_stopping_patience=0, feature_scaling='none')
            output = root / 'run'
            with redirect_stdout(io.StringIO()):
                train_classifier(config, output, device_name='cpu', cpu_threads=1, stop_after_steps=1)
            saved = json.loads((output / 'config.json').read_text())
            saved.pop('feature_scaling')
            (output / 'config.json').write_text(json.dumps(saved))
            checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
            checkpoint['config'].pop('feature_scaling')
            checkpoint.pop('feature_transform')
            torch.save(checkpoint, output / 'checkpoint.pt')
            artifact = torch.load(output / 'best_classifier.pt', weights_only=True)
            artifact.pop('feature_transform')
            torch.save(artifact, output / 'best_classifier.pt')
            with redirect_stdout(io.StringIO()):
                result = train_classifier(config_from_dict(saved), output, device_name='cpu', cpu_threads=1, resume=True)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['feature_scaling'], 'none')

    def test_prediction_uses_saved_feature_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'classifier').mkdir()
            vectors = torch.nn.functional.normalize(torch.tensor([[1., -.001, 0.], [1., .001, 0.]]), dim=1)
            transform = {'mode': 'standard', 'mean': torch.tensor([1., 0., 0.]),
                         'scale': torch.tensor([1., .001, 1.])}
            head = torch.nn.Linear(3, 2)
            with torch.no_grad():
                head.weight.copy_(torch.tensor([[0., -1., 0.], [0., 1., 0.]]))
                head.bias.zero_()
            torch.save({'head': head.state_dict(), 'embedding_dimension': 3,
                        'label_to_id': LABELS_BY_TASK['time_label'], 'feature_transform': transform},
                       root / 'classifier/best_classifier.pt')
            (root / 'classifier/config.json').write_text(json.dumps({'max_seq_length': 32, 'precision': 'fp32'}))
            with patch('wcb_4090.predict.SentenceTransformer'), patch('wcb_4090.predict.encode_sentences', return_value=vectors):
                results = predict(root, ['past', 'future'])
            expected = torch.softmax(head(transform_features(vectors, transform)), dim=1)
            for i, result in enumerate(results):
                for name, label in LABELS_BY_TASK['time_label'].items():
                    self.assertAlmostEqual(result['scores'][name], expected[i, label].item(), places=6)

    def test_collapsed_triplets_have_margin_loss_and_no_satisfied_triplets(self):
        embeddings = torch.ones(8, 3)
        labels = torch.tensor([0] * 4 + [1] * 4)
        loss = make_loss(torch.nn.Identity(), .2).compute_loss_from_embeddings([embeddings], labels)
        self.assertAlmostEqual(loss.item(), .2, places=5)
        geometry = batch_geometry(embeddings, labels, .2)
        self.assertAlmostEqual(geometry['mean_pairwise_cosine'], 1., places=5)
        self.assertEqual(geometry['triplet_satisfied_fraction'], 0.)

    def test_all_three_stages_and_portable_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            for task in LABELS_BY_TASK:
                data(root / 'data', task)
            args = ['--output', str(root / 'run'), '--model-path', str(model), '--data-root', str(root / 'data'),
                    '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1', '--batch-size', '8',
                    '--max-seq-length', '32', '--triplet-epochs', '1', '--triplet-eval-every', '1',
                    '--classifier-epochs', '2', '--embedding-batch-size', '4']
            self.assertEqual(pipeline(args), 0)
            summary = json.loads((root / 'run/pipeline_summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(len(summary['stages']), 6)
            self.assertEqual(set(summary['test_results']), set(LABELS_BY_TASK))
            for task in LABELS_BY_TASK:
                result = predict(root / 'run' / task, ['bank future policy'], 'cpu')
                self.assertIn(result[0]['label'], LABELS_BY_TASK[task])
                self.assertAlmostEqual(sum(result[0]['scores'].values()), 1., places=5)
            # Portable predictor finds the sibling encoder after the run is moved.
            (root / 'run').rename(root / 'moved')
            result = predict(root / 'moved/time_label', ['bank future policy'], 'cpu')
            self.assertEqual(len(result), 1)

    def test_cloud_triplet_only_then_local_cpu_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            tiny_encoder(model)
            data(root / 'data', 'time_label')
            args = ['--output', str(root / 'run'), '--model-path', str(model), '--data-root', str(root / 'data'),
                    '--tasks', 'time_label', '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
                    '--batch-size', '8', '--max-seq-length', '32', '--triplet-epochs', '1', '--triplet-only']
            self.assertEqual(pipeline(args), 0)
            self.assertFalse((root / 'run/time_label/classifier').exists())
            local_args = ['--run', str(root / 'run'), '--data-root', str(root / 'data'), '--tasks', 'time_label',
                          '--device', 'cpu', '--cpu-threads', '1', '--epochs', '2', '--embedding-batch-size', '4']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(local_classifiers(local_args), 0)
                self.assertEqual(local_classifiers(local_args + ['--resume']), 0)
            summary = json.loads((root / 'run/local_classifiers_summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['tasks']['time_label']['test']['test_rows'], 8)
            saved = json.loads((root / 'run/time_label/classifier/config.json').read_text())
            self.assertEqual(saved['max_seq_length'], 32)


if __name__ == '__main__':
    unittest.main()
