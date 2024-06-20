import logging
import os.path
import urllib.request

from floral_oracle.utils.paths import MODEL_PATH, MODEL_URL


def download_llama_cpp_model(url: str = MODEL_URL, destination: str = MODEL_PATH):
    logging.info("Downloading file from:")
    logging.info(url)
    urllib.request.urlretrieve(url, filename=destination)


def model_is_download() -> bool:
    return os.path.exists(MODEL_PATH)


def validate_model() -> None:
    if not model_is_download():
        download_llama_cpp_model()
