from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from floral_oracle.paths import MODEL_PATH
from floral_oracle.model import validate_model
from floral_oracle.corpus import load_corpus, split_corpus

prompt = hub.pull("rlm/rag-prompt-llama")

if __name__ == "__main__":
    validate_model()

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0,
        max_new_tokens=256,
        context_window=3900,
        n_gpu_layers=1,
        verbose=True,
    )

    corpus = load_corpus()
    documents: list[Document] = split_corpus(corpus)

    vector_store = FAISS.from_documents(
        documents, embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=dict(k=3, score_threshold=0.6),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs=dict(prompt=prompt),
    )

    result = qa_chain({"query": "How do I control aphids?"})
    result["result"]
