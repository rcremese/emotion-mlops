# train.py

import mlflow
import mlflow.pytorch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from emotion_mlops.models.emotion_classifier import EmotionClassifier
from emotion_mlops.data.datamodule_fer2013 import FER2013DataModule


def train_one_run(backbone: str, lr: float, batch_size: int):    
    TRACKING_URI = "file:///home/robin_cremese/Projects/Python/emotion-mlops/ml-runs"
    mlf_logger = MLFlowLogger(experiment_name="fer2013_emotion", tracking_uri=TRACKING_URI)

    dm = FER2013DataModule(batch_size=batch_size)
    model = EmotionClassifier(backbone=backbone, lr=lr)

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        logger=mlf_logger,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train_one_run(
        backbone="resnet18",
        lr=1e-3,
        batch_size=64
    )
