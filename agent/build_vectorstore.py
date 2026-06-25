from pathlib import Path

from decouple import config
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

text = Path("data/salon_knowledge.txt").read_text()

documents = [
    Document(page_content=text)
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    api_key=config("OPENAI_API_KEY")
)

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)


vectorstore.save_local("vector_db")

print("FAISS Index Saved Successfully")