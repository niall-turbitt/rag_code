# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Chain demo - Python (without Langchain)
# MAGIC
# MAGIC RAG chain implemented in pure Python, using Databricks Vector Search and Databricks Foundation Model APIs.
# MAGIC
# MAGIC ## RAG chain:
# MAGIC
# MAGIC #### Target use case
# MAGIC - Multi-turn Q&A on proprietary knowledge base
# MAGIC - Retrieval for unstructured data in a vector index
# MAGIC
# MAGIC #### Data source 
# MAGIC - Raw unstructured documents in UC volume (e.g, PDF files)
# MAGIC
# MAGIC ### RAG chain
# MAGIC - Inputs
# MAGIC   - User query string
# MAGIC   - Chat history (in OpenAI chat messages format)
# MAGIC - Retrieves the top K chunks from a Databricks Vector Search index
# MAGIC - Augment LLM prompt with retrieved chunks
# MAGIC - Output generation
# MAGIC - Outputs
# MAGIC   - OpenAI chat completions format

# COMMAND ----------

# DBTITLE 1,Databricks AI Stack Setup
# MAGIC %pip install --upgrade --quiet databricks-vectorsearch databricks-genai-inference

# COMMAND ----------

# DBTITLE 1,Python Environment Restart Handler
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

# DBTITLE 1,Databricks AI Search Integration
from typing import List, Dict, Any
import yaml

from databricks_genai_inference import ChatCompletion
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md
# MAGIC ### Env vars

# COMMAND ----------

# DBTITLE 1,Configuration Parameters
# Load configuration from YAML file
with open("3_rag_chain_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)

# Vector Search params
VS_ENDPOINT_NAME = rag_config["vector_search_endpoint_name"]
VS_INDEX_NAME = rag_config["vector_search_index"]
VS_COLUMNS = rag_config["vector_search_schema"]["chunk_text"]
NUM_RESULTS = rag_config["vector_search_parameters"]["num_results"]

# LLM params
LLM = rag_config["chat_endpoint"]
TEMPERATURE = rag_config["chat_endpoint_parameters"]["temperature"]
MAX_TOKENS = rag_config["chat_endpoint_parameters"]["max_tokens"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search component

# COMMAND ----------

# DBTITLE 1,Vector Search Function
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
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME
    )
    
    results = vs_index.similarity_search(
        query_text=query_text,
        columns=[VS_COLUMNS],
        num_results=num_results
    )
    return results


def format_docs(search_results: Dict[str, Any]) -> str:
    """Format the retrieved results into a single string.

    Args:
        search_results: Dictionary representing the search results.

    Returns:
        Single string containing the formatted documents.
    """
    chunk_template = rag_config["chunk_template"]
    docs = [row[0] for row in search_results["result"]["data_array"]]
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
    prompt_template = rag_config["chat_prompt_template"]
    prompt = prompt_template.format(question=question, context=context)
    
    response = ChatCompletion.create(
        model=LLM,
        messages=[
            {"role": "system", "content": "You are an assistant for question-answering tasks."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return response.message

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG chain

# COMMAND ----------

# DBTITLE 1,Contextual Answer Generation Routine
def rag_chain(question: str, num_results: int = 3) -> str:
    """Orchestrate the RAG chain.

    Args:
        question: User question
        num_results: Number of context documents to retrieve.

    Returns:
        Generated answer from the RAG chain.
    """
    search_results = similarity_search(question, num_results)
    context = format_docs(search_results)
    answer = generate_answer(question, context)
    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query chain

# COMMAND ----------

question = "What is ARES?"
answer = rag_chain(question, num_results=NUM_RESULTS)

print(f"Question: {question}")
print(f"Answer: {answer}")