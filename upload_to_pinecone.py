import hashlib
import json
import math
import os
import time
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag-chunks")

CHUNKS_PATH = "processed_chunks.json"
NAMESPACE = "portfolio-rag"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Could not find {CHUNKS_PATH}")


def first_existing(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def get_chunk_text(item: dict[str, Any]) -> str:
    text = first_existing(
        item,
        [
            "text",
            "chunk_text",
            "content",
            "page_content",
            "summary",
            "raw_text",
        ],
    )

    if not text:
        raise ValueError(
            f"Chunk is missing text content. Available keys: {list(item.keys())}"
        )

    return str(text)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def stable_field(item: dict[str, Any], key: str) -> Any:
    if key in item and item[key] is not None:
        return item[key]

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)

    return None


def make_stable_id(item: dict[str, Any], text: str) -> str:
    """
    Always create a stable Pinecone ID from deterministic chunk content.

    This guarantees that running the script twice with the same text
    produces the same vector IDs, so Pinecone overwrites instead of duplicating.
    """
    stable_identity = {
        "text": normalize_text(text),
        "source": stable_field(item, "source"),
        "filename": stable_field(item, "filename"),
        "file_name": stable_field(item, "file_name"),
        "page": stable_field(item, "page"),
        "page_number": stable_field(item, "page_number"),
        "section_name": stable_field(item, "section_name"),
        "chunk_index": stable_field(item, "chunk_index"),
    }
    stable_identity = {
        key: str(value)
        for key, value in stable_identity.items()
        if value is not None
    }
    stable_payload = json.dumps(
        stable_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()


def sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        if all(isinstance(x, str) for x in value):
            return value
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def build_metadata(item: dict[str, Any], text: str) -> dict[str, Any]:
    excluded_keys = {
        "embedding",
        "vector",
        "values",
    }

    metadata = {}

    for key, value in item.items():
        if key in excluded_keys:
            continue

        cleaned = sanitize_metadata_value(value)

        if cleaned is not None:
            metadata[key] = cleaned

    metadata["text"] = text
    metadata["embedding_model"] = EMBEDDING_MODEL_NAME

    return metadata


with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not isinstance(chunks, list):
    raise ValueError("processed_chunks.json must contain a list of chunk objects.")

print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

texts = [get_chunk_text(chunk) for chunk in chunks]

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    index = pc.index(PINECONE_INDEX_NAME)
except AttributeError:
    index = pc.Index(PINECONE_INDEX_NAME)

vectors = []

for index_number, (chunk, text, embedding) in enumerate(
    zip(chunks, texts, embeddings),
    start=1,
):
    vector_id = make_stable_id(chunk, text)
    vector_values = embedding.tolist()

    if len(vector_values) != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION}-dimensional vector, got {len(vector_values)}"
        )

    metadata = build_metadata(chunk, text)

    print(f"Vector ID {index_number}: {vector_id}")

    vectors.append(
        {
            "id": vector_id,
            "values": vector_values,
            "metadata": metadata,
        }
    )

print(f"Prepared {len(vectors)} vectors for upload.")

vector_ids = [vector["id"] for vector in vectors]

print(f"Deleting existing vectors in namespace '{NAMESPACE}' before upload...")
try:
    index.delete(
        ids=vector_ids,
        namespace=NAMESPACE,
    )
    time.sleep(5)
except Exception as error:
    if "Namespace not found" not in str(error):
        raise
    print(f"Namespace '{NAMESPACE}' does not exist yet; skipping delete.")

response = index.upsert(
    vectors=vectors,
    namespace=NAMESPACE,
)

print("Upload response:", response)

stats = index.describe_index_stats()
print("Index stats after upload:")
print(stats)
