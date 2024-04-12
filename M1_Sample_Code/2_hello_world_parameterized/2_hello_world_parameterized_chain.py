# Databricks notebook source
# DBTITLE 1,Install RAG Studio packages
# MAGIC %run ../wheel_installer

# COMMAND ----------

# Before logging this chain using the driver notebook, you need to comment out this line.
dbutils.library.restartPython() 

# COMMAND ----------

from operator import itemgetter

from databricks import rag 

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# COMMAND ----------

############
# Private Preview Feature MLflow Tracing
# RAG Studio dependes on MLflow to show a trace of your chain. The trace can help you easily debug your chain and keep track of inputs & responses your chain performs well or performs poorly.
############

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

# DBTITLE 1,Hello World Model
############
# RAG Studio requires your chain to accept an array of OpenAI-formatted messages as a `messages` parameter. Schema: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# These helper functions help parse the `messages` array
############

# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


############
# Get the configuration YAML
############
rag_config = rag.RagConfig("configs/2_hello_world_config.yaml")

############
# Fake model for this hello world example.
############
def fake_model(input):
    return f"Config: {rag_config.get('sample_param')}.  You asked `{input.get('user_query')}`. Conversation history: {input.get('chat_history')}"


############
# Simplest chain example
############
# RAG Studio requires the chain to return a string value.
chain = (
    {
        "user_query": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | RunnableLambda(fake_model)
    | StrOutputParser()
)

############
# You can test this chain locally in the notebook
############
question = {
    "messages": [
        {
            "role": "user",
            "content": "question 1",
        },
        {
            "role": "assistant",
            "content": "answer 1",
        },
        {
            "role": "user",
            "content": "new question!!",
        },
    ]
}

chain.invoke(question)

# COMMAND ----------

# You need to call `set_chain` in order for RAG Studio to log your chain.
rag.set_chain(chain)
