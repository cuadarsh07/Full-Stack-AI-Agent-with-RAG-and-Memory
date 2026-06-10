import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag-chunks")
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")

if not api_key:
    raise ValueError("Missing PINECONE_API_KEY in .env")

pc = Pinecone(api_key=api_key)

existing_indexes = pc.indexes.list().names()

if index_name not in existing_indexes:
    pc.indexes.create(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=cloud,
            region=region,
        ),
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Pinecone index already exists: {index_name}")

desc = pc.indexes.describe(index_name)
print("Index name:", desc.name)
print("Dimension:", desc.dimension)
print("Metric:", desc.metric)
print("Ready:", desc.status.ready)