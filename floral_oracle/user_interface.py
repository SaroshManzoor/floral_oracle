from time import sleep

import panel as pn
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from floral_oracle.chain import get_retrieval_chain

pn.extension()

conversational_chain: ConversationalRetrievalChain = get_retrieval_chain()


def get_response_from_chain(contents, user, instance):
    response = conversational_chain.invoke(contents)

    answer = response["answer"]

    for index in range(len(answer)):
        yield answer[0 : index + 1]
        sleep(0.005)


chat_bot = pn.chat.ChatInterface(
    callback=get_response_from_chain, max_height=500, callback_exception="verbose"
)

chat_bot.send(
    "Ask me about gardening & plants!",
    user="Assistant",
    respond=False,
)
chat_bot.servable()
