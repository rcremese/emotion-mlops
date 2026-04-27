# src/models/emotion_classifier.py

import torch
import lightning as L
import torch.nn as nn
import torchmetrics
import timm


class EmotionClassifier(L.LightningModule):
    def __init__(
        self, backbone: str, lr: float = 1e-3, in_chans: int = 3, num_classes: int = 7
    ):
        super().__init__()
        self.save_hyperparameters()

        try:
            self.model = timm.create_model(
                backbone, pretrained=True, num_classes=num_classes, in_chans=in_chans
            )
        except TypeError:
            raise ValueError(
                f"The {backbone} model does not support in_chans=1."
                "Choose a CNN backbone (resnet, efficientnet, convnext, ...)"
            )

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.precision = torchmetrics.Precision(
            "multiclass", num_classes=num_classes, average="weighted"
        )
        self.recall = torchmetrics.Recall(
            "multiclass", num_classes=num_classes, average="weighted"
        )
        self.f1_score = torchmetrics.F1Score(
            "multiclass", num_classes=num_classes, average="weighted"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)
        prec = self.precision(logits, labels)
        rec = self.recall(logits, labels)
        f1 = self.f1_score(logits, labels)

        self.log_dict(
            {"val/loss": loss, "val/accuracy": acc},
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {"val/precision": prec, "val/recall": rec, "val/f1_score": f1},
            on_epoch=True,
            logger=True,
            on_step=False,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams_initial.lr)
