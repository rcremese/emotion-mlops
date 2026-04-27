import mlflow
import mlflow.pytorch as mlfp
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
# from lightning.pytorch.callbacks import ModelCheckpoint

from emotion_mlops.models.emotion_classifier import EmotionClassifier
from emotion_mlops.data.datamodule_fer2013 import FER2013DataModule
from emotion_mlops.utils import TRACKING_URI


def train_one_run(
    backbone: str, lr: float, batch_size: int, nb_epochs: int, run_id: str | None = None
):
    dm = FER2013DataModule(batch_size=batch_size)
    model = EmotionClassifier(backbone=backbone, lr=lr, in_chans=1)
    tags = {"data_augmentation": "RandRotate+ColorJitter"}

    mlf_logger = MLFlowLogger(
        experiment_name="fer2013_emotion",
        tags=tags,
        tracking_uri=TRACKING_URI,
        log_model=False,
        run_id=run_id,
    )
    if run_id is None:
        run_id = mlf_logger.run_id

    # checkpoint = ModelCheckpoint(
    #     dirpath=f"{PROJECT_ROOT}/models",
    #     filename="best",
    #     save_last=True,
    #     save_top_k=1,
    #     monitor="val/acc",
    #     mode="max",
    # )

    trainer = Trainer(
        max_epochs=nb_epochs,
        accelerator="gpu",
        devices=1,
        logger=mlf_logger,
        # callbacks=[checkpoint],
    )

    mlfp.autolog()

    with mlflow.start_run(run_id=run_id):
        trainer.fit(model, dm)

    # mlflow.register_model(model_uri=f"runs:/{mlf_logger.run_id}/model", name=backbone)


if __name__ == "__main__":
    train_one_run(backbone="resnet18", lr=1e-3, batch_size=128, nb_epochs=20)
