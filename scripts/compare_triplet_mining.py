"""Controlled train/val-only ablation of mining and reference preservation."""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import BatchHardTripletLoss, BatchSemiHardTripletLoss

from wcb_4090.audit import embedding_geometry, linear_probe
from wcb_4090.batching import BalancedBatchSampler, load_rows
from wcb_4090.classifier import encode_sentences
from wcb_4090.train import atomic_json, encode_batch
from wcb_4090.triplet import cosine_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('runs/triplet-stability/ablation.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda')
    source = Path('data/processed_4090/wcb_time_5768')
    rows = {split: load_rows(source / f'{split}.parquet', split, count, seed)
            for split, count, seed in [('train', 512, 5768), ('val', 128, 5769)]}
    reference = {}
    for split in rows:
        cache = torch.load(f'runs/training-audit/original-gte-time/embeddings_{split}.pt', weights_only=True)
        lookup = {sid: i for i, sid in enumerate(cache['sentence_ids'])}
        reference[split] = cache['embeddings'][[lookup[row['sentence_id']] for row in rows[split]]]
    labels = {split: torch.tensor([row['label'] for row in value]) for split, value in rows.items()}
    baseline = {split: {'embeddings': value, 'labels': labels[split]} for split, value in reference.items()}
    report = {'steps': args.steps, 'test_split_used': False,
              'sampling': '512 per training class and 128 per validation class; seeds 5768/5769',
              'baseline_geometry': embedding_geometry(reference['val'], labels['val']),
              'baseline_probe': linear_probe(baseline['train'], baseline['val'], 'standard'), 'methods': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mining, weight in [('hard', 0.), ('semi_hard', 0.), ('semi_hard', 1.)]:
        name = f'{mining}_reference_{weight:g}'
        torch.manual_seed(5768); torch.cuda.manual_seed_all(5768)
        encoder = SentenceTransformer('models/gte-modernbert-base', device='cuda', local_files_only=True,
            model_kwargs={'attn_implementation': 'sdpa', 'dtype': torch.float32},
            config_kwargs={'reference_compile': False})
        encoder.max_seq_length = 256
        encoder[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        loss_class = BatchHardTripletLoss if mining == 'hard' else BatchSemiHardTripletLoss
        loss_fn = loss_class(encoder, distance_metric=cosine_distance, margin=.2)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-5, weight_decay=.01)
        warmup = int(args.steps * .1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: step / max(1, warmup) if step < warmup else max(0., (args.steps-step)/max(1,args.steps-warmup)))
        sampler = BalancedBatchSampler(labels['train'].tolist(), 32, 5768)
        history = []
        step = 0
        while step < args.steps:
            for indices in sampler.batches(step // len(sampler)):
                encoder.train()
                features, y = encode_batch(encoder, rows['train'], indices, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    x = encoder(features)['sentence_embedding'].float()
                triplet = loss_fn.compute_loss_from_embeddings([x], y)
                anchor = (1-F.cosine_similarity(x, reference['train'][indices].to(device))).mean()
                loss = triplet + weight * anchor
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); step += 1
                if step in (1, 50, 100, args.steps):
                    encoder.eval()
                    vectors = torch.cat([encode_sentences(encoder, [row['sentence'] for row in rows['val'][i:i+64]], device, 'bf16')
                                         for i in range(0, len(rows['val']), 64)])
                    item = {'step': step, 'triplet_loss': triplet.detach().item(), 'reference_loss': anchor.detach().item(),
                            'validation': embedding_geometry(vectors, labels['val']),
                            'validation_reference_cosine': F.cosine_similarity(vectors, reference['val']).mean().item()}
                    history.append(item)
                    print(json.dumps({'method': name, **item}), flush=True)
                if step >= args.steps:
                    break
        encoder.eval()
        cached = {}
        for split in rows:
            vectors = torch.cat([encode_sentences(encoder, [row['sentence'] for row in rows[split][i:i+64]], device, 'bf16')
                                 for i in range(0, len(rows[split]), 64)])
            cached[split] = {'embeddings': vectors, 'labels': labels[split]}
        report['methods'][name] = {'history': history,
            'validation_probe': linear_probe(cached['train'], cached['val'], 'standard')}
        atomic_json(args.output, report)
        print(json.dumps({'method': name, 'validation_macro_f1': report['methods'][name]['validation_probe']['macro_f1']}), flush=True)
        del optimizer, scheduler, loss_fn, encoder, x, features, loss, triplet, anchor
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
