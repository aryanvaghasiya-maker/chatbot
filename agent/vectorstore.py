from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def get_retriever():
    load = TextLoader("/home/ip42/Documents/chatbot/data/salon_knowledge.txt",encoding="utf-8")

    documents = load.load()
    
    docs = splitter.split_documents(documents)


    embeddings = OpenAIEmbeddings(
        api_key=(OPENAI_API_KEY)
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
