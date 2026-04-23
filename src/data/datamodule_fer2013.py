import io
import zipfile
import boto3
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class FER2013DataModule(pl.LightningDataModule):
    def __init__(
        self,
        bucket_name: str = "emotion-mlops",
        zip_key: str = "datasets/fer2013.zip",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.zip_key = zip_key
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.s3 = boto3.client("s3")
        self.data_dir = Path("/tmp/fer2013")

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

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
            self.train_dataset, self.val_dataset  = random_split(fit_dataset, [0.8, 0.2])

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
