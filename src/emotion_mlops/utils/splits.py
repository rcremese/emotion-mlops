import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


def create_stratified_indexes(
    dataset: ImageFolder, val_ratio: float = 0.2, seed: int | None = None
):
    targets = torch.tensor(dataset.targets)
    rng = torch.Generator().manual_seed(seed) if (seed is not None) else None

    train_indices, val_indices = [], []

    for label in targets.unique():
        indices = (targets == label).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        split = int(len(indices) * val_ratio)
        val_indices.extend(indices[:split].tolist())
        train_indices.extend(indices[split:].tolist())
    return train_indices, val_indices


def create_stratified_split(
    dataset: ImageFolder, val_ratio: float = 0.2, seed: int | None = None
) -> tuple[Subset, Subset]:
    train_indices, val_indices = create_stratified_indexes(dataset, val_ratio, seed)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
