import io
import zipfile
import boto3
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as v2
from torchvision.datasets import ImageFolder


def stratified_split(dataset : ImageFolder, val_ratio : float = 0.2, seed : int | None = None):
    targets = torch.tensor(dataset.targets)
    
    train_indices, val_indices = [], []

    rng = torch.Generator().manual_seed(seed) if (seed is not None) else None 

    for label in targets.unique():
        indices = (targets == label).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        split = int(len(indices) * val_ratio)
        val_indices.extend(indices[:split].tolist())
        train_indices.extend(indices[split:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

class FER2013DataModule(L.LightningDataModule):
    def __init__(
        self,
        bucket_name: str = "emotion-mlops",
        zip_key: str = "datasets/fer2013.zip",
        batch_size: int = 64,
        num_workers: int = 4,
        seed : int | None = None
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.zip_key = zip_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.s3 = boto3.client("s3")
        self.data_dir = Path("/tmp/fer2013")

        self.transform = v2.Compose([v2.Resize((48,48)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5], std=[0.5])])

    def prepare_data(self):
        """Télécharge et extrait le dataset depuis S3 si nécessaire."""
        if self.data_dir.exists():
            return  # déjà extrait

        print("📥 Téléchargement du dataset depuis S3...")

        obj = self.s3.get_object(
            Bucket=self.bucket_name,
            Key=self.zip_key
        )

        with zipfile.ZipFile(io.BytesIO(obj["Body"].read())) as archive:
            archive.extractall(self.data_dir)

        print("📦 Extraction terminée dans /tmp/fer2013")

    def setup(self, stage=None):
        """Charge les datasets avec ImageFolder."""
        if stage in (None, "fit"):
            fit_dataset = ImageFolder(
                root=str(self.data_dir / "train"),
                transform=self.transform
            )
            self.train_dataset, self.val_dataset  = stratified_split(fit_dataset, val_ratio=0.2, seed=self.seed)

        if stage in (None, "test"):
            self.test_dataset = ImageFolder(
                root=str(self.data_dir / "test"),
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
