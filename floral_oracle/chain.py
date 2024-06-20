import logging
import os.path

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from floral_oracle.corpus import load_corpus, split_corpus
from floral_oracle.embed_model import get_embedding_model
from floral_oracle.model import validate_model
from floral_oracle.utils.paths import MODEL_PATH, VECTOR_STORE_PATH


def get_retrieval_chain() -> ConversationalRetrievalChain:
    validate_model()

    logging.info(f"Loading Model:  {os.path.basename(MODEL_PATH)}")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0,
        max_tokens=1024,
        n_gpu_layers=50,
        verbose=False,
        n_ctx=4096,
    )
    embedding_model = get_embedding_model(fast=True)

    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        corpus = load_corpus()
        documents: list[Document] = split_corpus(corpus)

        vector_store = FAISS.from_documents(documents, embedding=embedding_model)
        vector_store.save_local(VECTOR_STORE_PATH)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=dict(k=3, score_threshold=0.6),
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )


if __name__ == "__main__":
    qa_chain = get_retrieval_chain()

    chat_history = []

    result = qa_chain({"question": "how do I prune basil?"})

    source_documents = result["source_documents"]

    sources = []
    for document in source_documents:
        sources.append(
            f"Title: {document.metadata['title']} |    "
            f"Author: {document.metadata['author']}    |    "
            f"Page: {document.metadata['page']}\n\n"
        )
