import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag-chunks")

if not api_key:
    raise ValueError("Missing PINECONE_API_KEY in .env")

pc = Pinecone(api_key=api_key)

existing_indexes = pc.indexes.list().names()

if index_name in existing_indexes:
    pc.indexes.delete(name=index_name)
    print(f"Deleted Pinecone index: {index_name}")
else:
    print(f"Index does not exist: {index_name}")