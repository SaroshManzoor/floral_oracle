# from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from floral_oracle.utils.paths import CORPUS_DIR_PATH


def load_corpus() -> list[Document]:
    all_files = list(Path(CORPUS_DIR_PATH).rglob(pattern="*.pdf"))

    corpus = []

    for file_path in all_files:
        loader = PyMuPDFLoader(file_path.as_posix())
        corpus.extend(loader.load())

    return corpus


def split_corpus(corpus: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(corpus)
