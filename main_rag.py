import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import json
from pathlib import Path
from langchain import hub
from langchain_core.vectorstores import InMemoryVectorStore


# Loading the JSON
loader = JSONLoader(
    file_path='Capstruct Interview -data.json',
    jq_schema= r'.[]',
    text_content=False,
)
docs = loader.load()


# Loading environment variables
load_dotenv()
langsmith_key = os.getenv("LANGSMITH_KEY")
openai_key = os.getenv("OPENAI_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = langsmith_key
os.environ["OPENAI_API_KEY"] = openai_key

# LLM Model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
prompt = hub.pull("rlm/rag-prompt")

# Vector Store
vector_store = InMemoryVectorStore(embeddings)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# Invoke
result = graph.invoke({"question": "What is the combined area of all the walls?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')