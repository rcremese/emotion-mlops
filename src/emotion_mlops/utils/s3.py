from pathlib import Path
import logging
import zipfile
import boto3
import io

def download_zip_from_s3(s3_uri: str, local_dir: Path):
    """
    Télécharge un fichier ZIP depuis S3 et le décompresse dans local_dir.
    Exemple s3_uri : 's3://emotion-mlops/datasets/fer2013.zip'
    """
    assert s3_uri.startswith("s3://")

    # Parse l'URI S3
    _, bucket, *key_parts = s3_uri.replace("s3://", "").split("/")
    key = "/".join(key_parts)

    s3 = boto3.client("s3")

    # Crée le dossier local si nécessaire
    local_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"📥 Téléchargement du ZIP : s3://{bucket}/{key}")
    # Téléchargement en mémoire
    obj = s3.get_object(Bucket=bucket, Key=key)
    zip_bytes = obj["Body"].read()

    # Décompression
    logging.info(f"📦 Extraction dans : {local_dir}")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(local_dir)
    logging.info("📦 Dataset extrait avec succès.")
