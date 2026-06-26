from decouple import config

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


COLLECTION_NAME = "salon_knowledge"

loader = TextLoader(
    "data/salon_knowledge.txt",
    encoding="utf-8",
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = splitter.split_documents(documents)



embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config("OPENAI_API_KEY")
)

client = QdrantClient(
    url=config("QDRANT_URL"),
    api_key=config("QDRANT_API_KEY"),
)

QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    url=config("QDRANT_URL"),
    api_key=config("QDRANT_API_KEY"),
    collection_name=COLLECTION_NAME,
)

print("✅ Vector Store uploaded successfully!")