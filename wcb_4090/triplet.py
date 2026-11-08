"""Cosine triplet mining and preservation of the source encoder's directions."""

import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers.sentence_transformer.losses import BatchHardTripletLoss, BatchSemiHardTripletLoss


def cosine_distance(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=1)
    return (1.0 - normalized @ normalized.T).clamp(min=0.0, max=2.0)


def validate_batch(labels: torch.Tensor):
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Labels must be a one-dimensional torch.long tensor")
    classes, counts = torch.unique(labels, return_counts=True)
    if len(classes) < 2 or (counts < 2).any():
        raise ValueError("A triplet batch needs >=2 classes and >=2 samples per class")


def make_loss(encoder: nn.Module, margin: float = 0.2, mining: str = "hard"):
    if not 0 < margin < 2:
        raise ValueError("Use a cosine-distance margin strictly between 0 and 2")
    losses = {"hard": BatchHardTripletLoss, "semi_hard": BatchSemiHardTripletLoss}
    if mining not in losses:
        raise ValueError("triplet-mining must be hard or semi_hard")
    loss = losses[mining](model=encoder, distance_metric=cosine_distance, margin=margin)
    # Both loss classes expose this under different names.
    loss.triplet_margin = margin
    return loss


def reference_alignment(embeddings: torch.Tensor, reference: torch.Tensor):
    """Half squared distance of unit vectors equals 1-cosine, without cancellation."""
    if embeddings.shape != reference.shape:
        raise ValueError("Reference and student embeddings must have the same shape")
    student = F.normalize(embeddings.float(), p=2, dim=1)
    teacher = F.normalize(reference.detach().float(), p=2, dim=1)
    return .5 * (student - teacher).square().sum(1).mean()


def hardest_indices(embeddings: torch.Tensor, labels: torch.Tensor):
    """Inspect selected triplets without changing the library's training loss."""
    validate_batch(labels)
    distances = cosine_distance(embeddings)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_indices = distances.masked_fill(~positive, -torch.inf).argmax(dim=1)
    negative_indices = distances.masked_fill(same, torch.inf).argmin(dim=1)
    return positive_indices, negative_indices


@torch.no_grad()
def batch_geometry(embeddings: torch.Tensor, labels: torch.Tensor, margin: float):
    """A loss near the margin can conceal vectors collapsing to one direction."""
    distances = cosine_distance(embeddings.float())
    positive, negative = hardest_indices(embeddings.float(), labels)
    anchors = torch.arange(len(labels), device=labels.device)
    positive_distances = distances[anchors, positive]
    negative_distances = distances[anchors, negative]
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return {
        "mean_pairwise_cosine": (1 - distances[off_diagonal]).mean().item(),
        "hard_positive_distance": positive_distances.mean().item(),
        "hard_negative_distance": negative_distances.mean().item(),
        "triplet_satisfied_fraction": ((negative_distances - positive_distances) >= margin).float().mean().item(),
    }
