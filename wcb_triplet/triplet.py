"""Cosine-distance wiring for the library's supervised BatchHardTripletLoss."""

import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers.sentence_transformer.losses import BatchHardTripletLoss


def cosine_distance(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=1)
    return (1.0 - normalized @ normalized.T).clamp(min=0.0, max=2.0)


def validate_batch(labels: torch.Tensor):
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Labels must be a one-dimensional torch.long tensor")
    classes, counts = torch.unique(labels, return_counts=True)
    if len(classes) < 2 or (counts < 2).any():
        raise ValueError("A triplet batch needs >=2 classes and >=2 samples per class")


def make_loss(encoder: nn.Module, margin: float = 0.2) -> BatchHardTripletLoss:
    if not 0 < margin < 2:
        raise ValueError("Use a cosine-distance margin strictly between 0 and 2")
    return BatchHardTripletLoss(
        model=encoder, distance_metric=cosine_distance, margin=margin
    )


def hardest_indices(embeddings: torch.Tensor, labels: torch.Tensor):
    """Inspect selected triplets without changing the library's training loss."""
    validate_batch(labels)
    distances = cosine_distance(embeddings)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_indices = distances.masked_fill(~positive, -torch.inf).argmax(dim=1)
    negative_indices = distances.masked_fill(same, torch.inf).argmin(dim=1)
    return positive_indices, negative_indices
