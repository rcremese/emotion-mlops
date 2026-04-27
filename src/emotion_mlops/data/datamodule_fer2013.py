import logging
from pathlib import Path
from emotion_mlops.utils import (
    create_stratified_indexes,
    PROJECT_ROOT,
    download_zip_from_s3,
)

import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


class FER2013DataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | None = None,
        s3_path: str = "s3://emotion-mlops/datasets/fer2013.zip",
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int | None = None,
    ):
        super().__init__()
        self.root = (
            Path(root) if root else PROJECT_ROOT.joinpath("data", "raw", "fer2013")
        )
        self.s3_path = s3_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Grayscale(),
                v2.RandomCrop(48, padding=4),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(degrees=[-15, 15]),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.val_transforms = v2.Compose(
            [
                v2.Resize((48, 48)),
                v2.ToImage(),
                v2.Grayscale(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _local_data_available(self) -> bool:
        return self.root.exists() and any(self.root.iterdir())

    def prepare_data(self):
        """Télécharge et extrait le dataset depuis S3 si nécessaire."""
        if self._local_data_available():
            logging.info(f"[DataModule] Dataset local trouvé : {self.root}")
        else:
            logging.warning(
                f"[DataModule] Dataset local absent → téléchargement depuis S3 : {self.s3_path}"
            )
            self.root.mkdir(parents=True, exist_ok=True)
            download_zip_from_s3(self.s3_path, self.root)

    def setup(self, stage=None):
        """Charge les datasets avec ImageFolder."""
        if stage in (None, "fit"):
            train_dataset = ImageFolder(
                root=self.root.joinpath("train"), transform=self.train_transforms
            )
            val_dataset = ImageFolder(
                root=self.root.joinpath("train"), transform=self.val_transforms
            )

            # fit_dataset = ImageFolder(root=self.root.joinpath("train"))
            train_indx, val_indx = create_stratified_indexes(
                train_dataset, val_ratio=0.2, seed=self.seed
            )
            self.train_dataset = Subset(train_dataset, train_indx)
            self.val_dataset = Subset(val_dataset, val_indx)

        if stage in (None, "test"):
            self.test_dataset = ImageFolder(
                root=self.root.joinpath("test"), transform=self.val_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
