from langchain import hub
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from floral_oracle.corpus import load_corpus, split_corpus
from floral_oracle.embed_model import get_embedding_model
from floral_oracle.model import validate_model
from floral_oracle.paths import MODEL_PATH

prompt = hub.pull("rlm/rag-prompt-llama")

if __name__ == "__main__":
    validate_model()

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0,
        max_tokens=1024,
        n_gpu_layers=2,
        verbose=False,
        n_ctx=4096
    )

    corpus = load_corpus()
    documents: list[Document] = split_corpus(corpus)

    vector_store = FAISS.from_documents(
        documents, embedding=get_embedding_model(fast=True)
    )

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

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        # prompt=prompt,
        memory=memory,
        return_source_documents=True,
    )

    chat_history = []

    result = qa_chain(
        {"question": "How to get rid of mealy-bugs?", "chat_history": chat_history}
    )
    chat_history.extend(result["chat_history"])

    result = qa_chain(
        {"question": "what about fungus gnats", "chat_history": chat_history}
    )

    chat_history.extend(result["chat_history"])

    result = qa_chain(
        {"question": "are they harmful to plants?", "chat_history": chat_history}
    )

    answer = result["answer"].strip()

    #
    # # chat
    #
    # answer += "\n\n\nSource(s): \n"
    #
    # for document in result["source_documents"][:1]:
    #     answer += f"Title: {document.metadata['title']}\n"
    #     answer += f"Author: {document.metadata['author']}\n"
    #     answer += f"Page: {document.metadata['page']}\n"
    #
    print(answer)
