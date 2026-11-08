"""Evaluate frozen saved models on other data splits, exposing train/val overlap."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path

import torch
from torch import nn

from .batching import load_rows
from .classifier import config_from_dict, encoder_hashes, make_embeddings, metrics, transform_features
from .prepare import prepare, text_key
from .tasks import LABELS_BY_TASK
from .train import atomic_json, resolve_path, sha256


def overlap_cohorts(rows, train, validation):
    """Check both source row identity and whitespace-equivalent sentence content."""
    train_ids = {row['sentence_id'] for row in train}
    train_texts = {text_key(row['sentence']) for row in train}
    val_ids = {row['sentence_id'] for row in validation}
    val_texts = {text_key(row['sentence']) for row in validation}
    cohorts = {'all': list(range(len(rows))), 'seen_in_training': [],
               'seen_in_validation_only': [], 'unseen_in_training_and_validation': []}
    for index, row in enumerate(rows):
        key = text_key(row['sentence'])
        if row['sentence_id'] in train_ids or key in train_texts:
            cohorts['seen_in_training'].append(index)
        elif row['sentence_id'] in val_ids or key in val_texts:
            cohorts['seen_in_validation_only'].append(index)
        else:
            cohorts['unseen_in_training_and_validation'].append(index)
    return cohorts


def cohort_metrics(logits, labels, indices, classes):
    if not indices:
        return {'rows': 0, 'accuracy': None, 'macro_f1': None, 'weighted_f1': None}
    value = metrics(logits[indices], labels[indices], classes)
    weighted = sum(item['f1'] * item['support'] for item in value['per_class']) / len(indices)
    return {'rows': len(indices), **value, 'weighted_f1': weighted}


def model_fingerprint(directory, config):
    return {'classifier': sha256(directory / 'best_classifier.pt'),
            'classifier_config': sha256(directory / 'config.json'),
            'encoder': encoder_hashes(resolve_path(config.encoder_path))}


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/wcb-stable-all')
    parser.add_argument('--data-root', default='data/processed_4090')
    parser.add_argument('--seeds', nargs='+', choices=['5768', '78516', '944601'], default=['5768', '78516', '944601'])
    parser.add_argument('--tasks', nargs='+', choices=list(LABELS_BY_TASK), default=list(LABELS_BY_TASK))
    parser.add_argument('--output', default='runs/cross-seed-evaluation')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    root, output, data_root = resolve_path(args.run), resolve_path(args.output), resolve_path(args.data_root)
    output.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'evaluation_only', 'training_performed': False, 'model_run': str(root),
              'interpretation': 'These are the SAME 5768-trained models evaluated on different test partitions, not separately trained seed experiments.',
              'cohort_definition': 'Unseen excludes any sentence ID or whitespace-equivalent text in the model training or validation data.',
              'results': {}}
    for task in args.tasks:
        classifier = root / task / 'classifier'
        config = config_from_dict(json.loads((classifier / 'config.json').read_text()))
        before = model_fingerprint(classifier, config)
        source_data = resolve_path(config.data_dir)
        train = load_rows(source_data / 'train.parquet', 'train', None, config.seed, task)
        validation = load_rows(source_data / 'val.parquet', 'val', None, config.seed, task)
        artifact = torch.load(classifier / 'best_classifier.pt', map_location='cpu', weights_only=True)
        if artifact['label_to_id'] != LABELS_BY_TASK[task]:
            raise ValueError('Saved classifier task mapping changed')
        head = nn.Linear(artifact['embedding_dimension'], len(LABELS_BY_TASK[task])).eval()
        head.load_state_dict(artifact['head'])
        report['results'][task] = {'model_training_seed': config.seed, 'model_fingerprint': before, 'test_partitions': {}}
        for seed in args.seeds:
            data_dir = data_root / f"wcb_{task.removesuffix('_label')}_{seed}"
            case_output = output / seed / task
            case_output.mkdir(parents=True, exist_ok=True)
            if not all((data_dir / f'{split}.parquet').exists() for split in ('train', 'val', 'test')):
                with (case_output / 'preparation.log').open('w') as log, redirect_stdout(log):
                    prepare(seed, data_dir, task)
            rows = load_rows(data_dir / 'test.parquet', 'test', None, config.seed, task)
            config.data_dir = str(data_dir)
            # Reuse the existing test cache only when evaluating that exact original partition.
            original_test = source_data / 'test.parquet'
            if sha256(data_dir / 'test.parquet') == sha256(original_test):
                old_cache, new_cache = classifier / 'embeddings_test.pt', case_output / 'embeddings_test.pt'
                if old_cache.exists() and not new_cache.exists():
                    os.link(old_cache, new_cache)
            print(json.dumps({'event': 'evaluate', 'task': task, 'test_seed': seed, 'rows': len(rows),
                              'model_training_seed': config.seed, 'weights_frozen': True}), flush=True)
            cache = make_embeddings(config, case_output, {'test': rows}, torch.device(args.device))['test']
            logits = head(transform_features(cache['embeddings'], artifact.get('feature_transform')))
            labels = cache['labels']
            cohorts = overlap_cohorts(rows, train, validation)
            result = {'test_seed': seed, 'test_sha256': sha256(data_dir / 'test.parquet'),
                      'cohorts': {name: cohort_metrics(logits, labels, indices, len(LABELS_BY_TASK[task]))
                                  for name, indices in cohorts.items()}}
            # The original partition should reproduce the saved test scores without changing anything.
            if data_dir.resolve() == source_data.resolve():
                saved = json.loads((classifier / 'test_metrics.json').read_text())
                for metric in ('accuracy', 'macro_f1'):
                    if abs(result['cohorts']['all'][metric] - saved[metric]) > 1e-6:
                        raise ValueError(f'Original test metrics did not reproduce: {task}, {metric}')
            atomic_json(case_output / 'metrics.json', result)
            with (case_output / 'predictions.jsonl').open('w') as stream:
                for row, prediction in zip(rows, logits.argmax(1)):
                    stream.write(json.dumps({'sentence_id': row['sentence_id'], 'label': row['label'],
                                             'prediction': int(prediction)}) + '\n')
            report['results'][task]['test_partitions'][seed] = result
            atomic_json(output / 'summary.json', report)
            print(json.dumps({'task': task, 'test_seed': seed, 'cohort_scores': {
                name: {key: value[key] for key in ('rows', 'accuracy', 'macro_f1')}
                for name, value in result['cohorts'].items()}}), flush=True)
        if model_fingerprint(classifier, config) != before:
            raise ValueError('Model files changed during evaluation')
    report['status'] = 'completed'
    atomic_json(output / 'summary.json', report)
    print('Evaluation completed:', output / 'summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
