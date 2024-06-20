import panel as pn
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from floral_oracle.chain import get_retrieval_chain

pn.extension()

conversational_chain: ConversationalRetrievalChain = get_retrieval_chain()


def get_response_from_chain(contents, user, instance):
    response = conversational_chain.invoke(contents)

    answer = response["answer"]
    source_documents = response["source_documents"]

    rephrased_response_start = "Context:\n"

    if len(source_documents):
        answer += "\n\n\nSource(s): \n"

        sources = []
        for document in source_documents:
            sources.append(
                f"Title: {document.metadata['title']} | "
                f"Author: {document.metadata['author']} | "
                f"Page: {document.metadata['page']}\n\n"
            )

        answer += "".join(set(sources))

    else:
        answer = answer[: answer.find(rephrased_response_start)]

        answer = (
            f"Couldn't find anything related to your query in the data base\n"
            f"Here is what I think: \n\n\n" + answer
        )
    # # Stream
    # for index in range(len(answer)):
    #     yield answer[0 : index + 1]
    #     sleep(0.005)

    return answer


chat_bot = pn.chat.ChatInterface(
    callback=get_response_from_chain, max_height=500, callback_exception="verbose"
)

chat_bot.send(
    "Ask me about gardening & plants!",
    user="Assistant",
    respond=False,
)
chat_bot.servable()
