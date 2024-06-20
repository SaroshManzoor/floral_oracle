from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_core.embeddings import Embeddings

from floral_oracle.utils.device import get_device
from floral_oracle.utils.paths import MODEL_PATH


def get_embedding_model(fast=True) -> Embeddings:
    if fast:
        model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", model_kwargs=dict(device=get_device())
        )

    else:
        model = LlamaCppEmbeddings(model_path=MODEL_PATH, n_gpu_layers=2)

    return model
