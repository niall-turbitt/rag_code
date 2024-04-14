# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Chain demo - Python (without Langchain)
# MAGIC
# MAGIC RAG chain implemented in pure Python, using Databricks Vector Search and Databricks Foundation Model APIs.

# COMMAND ----------

# MAGIC %run ../../wheel_installer

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet databricks-vectorsearch databricks-genai-inference

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

from typing import List, Dict, Any

import yamlimport logging
from typing import List, Dict, Any

import yaml
from databricks import rag
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
import mlflow
from mlflow.deployments import get_deploy_client

# Constants
CHUNK_TEXT_COLUMN = "chunked_text"
DOCUMENT_SOURCE_COLUMN = "doc_uri"
PRIMARY_KEY_COLUMN = "chunk_id"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@mlflow.trace(name="similarity_search")
def similarity_search(query_text: str, num_results: int) -> List[Dict[str, Any]]:
    """Perform similarity search on the vector index."""
    vs_client = VectorSearchClient(disable_notice=True)
    vs_index = vs_client.get_index(
        endpoint_name=rag_config.get("vector_search_endpoint_name"),
        index_name=rag_config.get("vector_search_index"),
    )
    vector_search_schema = rag_config.get("vector_search_schema")

    rag.set_vector_search_schema(
        primary_key=vector_search_schema.get(PRIMARY_KEY_COLUMN),
        text_column=vector_search_schema.get(CHUNK_TEXT_COLUMN),
        doc_uri=vector_search_schema.get(DOCUMENT_SOURCE_COLUMN),
    )

    try:
        results = vs_index.similarity_search(
            query_text=query_text,
            columns=[
                vector_search_schema.get(PRIMARY_KEY_COLUMN),
                vector_search_schema.get(CHUNK_TEXT_COLUMN),
                vector_search_schema.get(DOCUMENT_SOURCE_COLUMN),
            ],
            num_results=num_results,
        )
        return results
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise


@mlflow.trace(name="format_docs")
def format_docs(search_results: List[List[Any]]) -> str:
    """Format the retrieved results into a single string."""
    chunk_template = rag_config.get("chunk_template")
    docs = [row[1] for row in search_results]
    formatted_docs = [chunk_template.format(chunk_text=doc) for doc in docs]
    return "".join(formatted_docs)


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str) -> str:
    """Generate an answer to the question using the provided context."""
    prompt_template = rag_config.get("chat_prompt_template")
    prompt = prompt_template.format(question=question, context=context)

    try:
        response = ChatCompletion.create(
            model=rag_config.get("chat_endpoint"),
            messages=[
                {"role": "system", "content": "You are an assistant for question-answering tasks."},
                {"role": "user", "content": prompt},
            ],
            **rag_config.get("chat_endpoint_parameters"),
        )
        return response.message
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        raise


@mlflow.trace(name="rag_chain")
def rag_chain(messages: List[Dict[str, str]], num_results: int = 3) -> str:
    """Orchestrate the RAG chain."""
    question = messages[-1]["content"]
    with mlflow.start_span(name="similarity_search_span", inputs={"question": question, "num_results": num_results}) as span:
        search_results = similarity_search(question, num_results)
        span.set_outputs({"search_results": search_results["result"]["data_array"]})

    with mlflow.start_span(name="format_docs_span") as span:
        context = format_docs(search_results["result"]["data_array"])
        span.set_inputs({"search_results": search_results["result"]["data_array"]})
        span.set_outputs({"context": context})

    with mlflow.start_span(name="generate_answer_span") as span:
        answer = generate_answer(question, context)
        span.set_inputs({"question": question, "context": context})
        span.set_outputs({"answer": answer})

    return answer


# Get the configuration YAML
rag_config = rag.RagConfig("config.yaml")

# Set the chain for RAG Studio
rag.set_chain(rag_chain)

# Deploy Client
deploy_client = get_deploy_client("databricks")

# Example usage
messages = [{"role": "user", "content": "What is the motivation behind ARES?"}]

# Autolog using MLflow
mlflow.langchain.autolog()

answer = rag_chain(messages, num_results=3)
print(f"Question: {messages[-1]['content']}")
print(f"Answer: {answer}")
from databricks import rag
from databricks_genai_inference import ChatCompletion
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md
# MAGIC ### Env vars

# COMMAND ----------

############
# Get the configuration YAML
############
rag_config = rag.RagConfig("config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search component

# COMMAND ----------

def similarity_search(query_text: str, num_results: int) -> List[Dict[str, Any]]:
    """Perform similarity search on the vector index.

    Args:
        query_text: User query
        num_results: Number of results to retrieve.

    Returns:
        List of dictionaries representing retrieved documents.
    """
    vs_client = VectorSearchClient(disable_notice=True)
    
    vs_index = vs_client.get_index(
        endpoint_name=rag_config.get("vector_search_endpoint_name"),
        index_name=rag_config.get("vector_search_index")
    )

    vector_search_schema = rag_config.get("vector_search_schema")

    ############
    # Required to:
    # 1. Enable the RAG Studio Review App to properly display retrieved chunks
    # 2. Enable evaluation suite to measure the retriever
    ############
    rag.set_vector_search_schema(
        primary_key=vector_search_schema.get("primary_key"),
        text_column=vector_search_schema.get("chunk_text"),
        doc_uri=vector_search_schema.get(
            "document_source"
        ),  # Review App uses `doc_uri` to display chunks from the same document in a single view
    )
    
    results = vs_index.similarity_search(
        query_text=query_text,
        columns=[
            vector_search_schema.get("primary_key"),
            vector_search_schema.get("chunk_text"),
            vector_search_schema.get("document_source"),
        ],
        num_results=num_results
    )
    return results


def format_docs(search_results: List[List[Any]]) -> str:
    """Format the retrieved results into a single string.

    Args:
        search_results: List of lists representing the search results.

    Returns:
        Single string containing the formatted documents.
    """
    chunk_template = rag_config.get("chunk_template")
    docs = [row[1] for row in search_results]  
    formatted_docs = [chunk_template.format(chunk_text=doc) for doc in docs]
    return "".join(formatted_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM component

# COMMAND ----------

# DBTITLE 1,Contextual Question Answering Function
def generate_answer(question: str, context: str) -> str:
    """Generate an answer to the question using the provided context.

    Args:
        question: User question
        context: Retrieved context documents.

    Returns:
        The generated answer.
    """
    prompt_template = rag_config.get("chat_prompt_template")
    prompt = prompt_template.format(question=question, context=context)
    
    response = ChatCompletion.create(
        model=rag_config.get("chat_endpoint"),
        messages=[
            {"role": "system", "content": "You are an assistant for question-answering tasks."},
            {"role": "user", "content": prompt}
        ],
        **rag_config.get("chat_endpoint_parameters")
    )
    return response.message

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG chain

# COMMAND ----------

def rag_chain(messages: List[Dict[str, str]], num_results: int = 3) -> str:
    """Orchestrate the RAG chain.

    Args:
        messages: List of dictionaries representing the chat messages.
        num_results: Number of context documents to retrieve.

    Returns:
        Generated answer from the RAG chain.
    """
    question = messages[-1]["content"]
    search_results = similarity_search(question, num_results)
    context = format_docs(search_results["result"]["data_array"])
    answer = generate_answer(question, context)
    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query chain

# COMMAND ----------

messages = [
    {
        "role": "user",
        "content": "What is the motivation behind ARES?"
    }
]

answer = rag_chain(messages, num_results=3)

print(f"Question: {messages[-1]['content']}")
print(f"Answer: {answer}")

# COMMAND ----------

############
# Tell RAG Studio about the chain - required for logging, but not local testing of this chain
############
rag.set_chain(rag_chain)

# COMMAND ----------

import logging
from typing import List, Dict, Any

import yaml
from databricks import rag
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
import mlflow
from mlflow.deployments import get_deploy_client

# Constants
CHUNK_TEXT_COLUMN = "chunked_text"
DOCUMENT_SOURCE_COLUMN = "doc_uri"
PRIMARY_KEY_COLUMN = "chunk_id"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@mlflow.trace(name="similarity_search")
def similarity_search(query_text: str, num_results: int) -> List[Dict[str, Any]]:
    """Perform similarity search on the vector index."""
    vs_client = VectorSearchClient(disable_notice=True)
    vs_index = vs_client.get_index(
        endpoint_name=rag_config.get("vector_search_endpoint_name"),
        index_name=rag_config.get("vector_search_index"),
    )
    vector_search_schema = rag_config.get("vector_search_schema")

    rag.set_vector_search_schema(
        primary_key=vector_search_schema.get(PRIMARY_KEY_COLUMN),
        text_column=vector_search_schema.get(CHUNK_TEXT_COLUMN),
        doc_uri=vector_search_schema.get(DOCUMENT_SOURCE_COLUMN),
    )

    try:
        results = vs_index.similarity_search(
            query_text=query_text,
            columns=[
                vector_search_schema.get(PRIMARY_KEY_COLUMN),
                vector_search_schema.get(CHUNK_TEXT_COLUMN),
                vector_search_schema.get(DOCUMENT_SOURCE_COLUMN),
            ],
            num_results=num_results,
        )
        return results
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise


@mlflow.trace(name="format_docs")
def format_docs(search_results: List[List[Any]]) -> str:
    """Format the retrieved results into a single string."""
    chunk_template = rag_config.get("chunk_template")
    docs = [row[1] for row in search_results]
    formatted_docs = [chunk_template.format(chunk_text=doc) for doc in docs]
    return "".join(formatted_docs)


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str) -> str:
    """Generate an answer to the question using the provided context."""
    prompt_template = rag_config.get("chat_prompt_template")
    prompt = prompt_template.format(question=question, context=context)

    try:
        response = ChatCompletion.create(
            model=rag_config.get("chat_endpoint"),
            messages=[
                {"role": "system", "content": "You are an assistant for question-answering tasks."},
                {"role": "user", "content": prompt},
            ],
            **rag_config.get("chat_endpoint_parameters"),
        )
        return response.message
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        raise


@mlflow.trace(name="rag_chain")
def rag_chain(messages: List[Dict[str, str]], num_results: int = 3) -> str:
    """Orchestrate the RAG chain."""
    question = messages[-1]["content"]
    with mlflow.start_span(name="similarity_search_span", inputs={"question": question, "num_results": num_results}) as span:
        search_results = similarity_search(question, num_results)
        span.set_outputs({"search_results": search_results["result"]["data_array"]})

    with mlflow.start_span(name="format_docs_span") as span:
        context = format_docs(search_results["result"]["data_array"])
        span.set_inputs({"search_results": search_results["result"]["data_array"]})
        span.set_outputs({"context": context})

    with mlflow.start_span(name="generate_answer_span") as span:
        answer = generate_answer(question, context)
        span.set_inputs({"question": question, "context": context})
        span.set_outputs({"answer": answer})

    return answer


# Get the configuration YAML
rag_config = rag.RagConfig("config.yaml")

# Set the chain for RAG Studio
rag.set_chain(rag_chain)

# Deploy Client
deploy_client = get_deploy_client("databricks")

# Example usage
messages = [{"role": "user", "content": "What is the motivation behind ARES?"}]

# Autolog using MLflow
mlflow.langchain.autolog()

answer = rag_chain(messages, num_results=3)
print(f"Question: {messages[-1]['content']}")
print(f"Answer: {answer}")

# COMMAND ----------


