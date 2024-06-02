from llama_index.core import VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (completion_to_prompt,
                                                    messages_to_prompt)

from llama_index.core.readers import StringIterableReader

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0,
    max_new_tokens=256,
    context_window=3900,
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

documents = [
    "Shamrock plants require direct sun for best growth and flowering",
    "They usually bloom all winter if placed in a bright sunny window",
    "These plants prefer soil that is kept barely moist and will do fine if the soil dries slightly between watering",
    "Oxalis plants should be fertilized only when the plant is actively growing.",
    "Shamrock plants like cooler temperatures, especially when in bloom. ",
    "These temperatures should be between 50-65 degrees F at night, and no greater than 75 degrees F during the day. ",
    "Temperatures above 75 degrees F may induce dormancy.",
    "In the summer months, Shamrocks should be allowed to go dormant. ",
    "The first sign that a plant is entering dormancy is leaf dieback.",
    "If this begins to occur, stop watering and fertilizing the plant. ",
    "The leaves can be cut back or allowed to die back on their own and the plant should be moved to a cool, dark place for two to three months.",
    "At the end of the dormant period, new foliage will begin emerging from the soil.",
    "This is a signal to move the plant to a sunny window and to begin watering and fertilization.",
    "If the oxalis plant is tall and lanky, it needs more light or may also occur if the conditions in the home are too warm.",
    "If your plant is not blooming, it probably needs a good rest. ",
    "Cut back on watering and fertilizing and let it go dormant.",
    "In two or three months, the plant will begin to grow again and should flower if it receives good care.",
    "A yellowing plant may be a sign you are watering it too much. Too little water and your plant will wilt.",
    "Shamrock plants are usually not bothered by insect pest, but are susceptible to root rot if kept too wet.",
]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(
    StringIterableReader().load_data(documents), embed_model=embed_model)

# set up query engine
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("what to do if my plant is leggy?")
print(response.response)