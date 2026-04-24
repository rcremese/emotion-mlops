import io
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image
import numpy as np

from emotion_mlops.data.datamodule_fer2013 import FER2013DataModule


def create_fake_zip():
    """Crée un ZIP minimal en mémoire avec quelques images factices."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as z:
        # Génère 4 images 48x48
        for split in ["train", "test"]:
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                z.writestr(f"{split}/happy/img_{split}_{i}.png", img_bytes.read())

    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def fake_s3_response():
    """Fixture qui mocke boto3.get_object pour renvoyer un ZIP factice."""
    fake_zip = create_fake_zip()

    return {
        "Body": io.BytesIO(fake_zip)
    }


@patch("boto3.client")
def test_datamodule_prepare_and_setup(mock_boto, fake_s3_response, tmp_path):
    """Teste que le DataModule télécharge, extrait et charge correctement les données."""

    # Mock boto3.get_object
    mock_s3 = mock_boto.return_value
    mock_s3.get_object.return_value = fake_s3_response

    # On force le datamodule à utiliser un dossier temporaire pytest
    dm = FER2013DataModule()
    dm.data_dir = tmp_path / "fer2013"

    # Étape 1 : prepare_data (téléchargement + extraction)
    dm.prepare_data()
    assert dm.data_dir.exists()

    # Étape 2 : setup (chargement des datasets)
    dm.setup("fit")

    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0

    # Étape 3 : test d’un batch
    images, labels = next(iter(dm.train_dataloader()))
    
    assert images.shape[0] == dm.batch_size or images.shape[0] < dm.batch_size
    assert images.shape[1:] == (3, 48, 48)
    assert labels.shape[0] == images.shape[0]
