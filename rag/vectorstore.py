from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from decouple import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader

COLLECTION_NAME = "salon_knowledge"

embeddings = OpenAIEmbeddings(api_key=config("OPENAI_API_KEY") )

client = QdrantClient(url=config("QDRANT_URL"),api_key=config("QDRANT_API_KEY"))

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3} )
  