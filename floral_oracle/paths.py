import os

import floral_oracle

BASE_PATH = os.path.dirname(os.path.dirname(floral_oracle.__file__))

MODEL_DIR_PATH = os.path.join(BASE_PATH, ".cache")
os.makedirs(MODEL_DIR_PATH, exist_ok=True)

CORPUS_DIR_PATH = os.path.join(BASE_PATH, "corpus")

MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR_PATH, os.path.basename(MODEL_URL))
