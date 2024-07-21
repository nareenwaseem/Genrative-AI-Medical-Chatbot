
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PC
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot-2"
docs_chunks = [t.page_content for t in text_chunks]
vectorstore = PineconeVectorStore.from_texts(
    docs_chunks,
    index_name=index_name,
    embedding=embeddings
)