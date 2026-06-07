import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from tavily import TavilyClient

load_dotenv()

EMBEDDING_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
DEFAULT_WEB_RESULTS = 3
DEFAULT_RESUME_RESULTS = 3
MAX_TOOL_RESULT_CHARS = 2500

hf_token = os.environ.get("HF_TOKEN")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

mcp = FastMCP("Adarsh Portfolio Server")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolErrorPayload(StrictBaseModel):
    type: str
    message: str


class ToolExecutionPayload(StrictBaseModel):
    success: bool
    tool_name: str
    query: str
    summary: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[ToolErrorPayload] = None


class BulletproofHFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not configured.")

        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        last_error = "Unknown error"

        for attempt in range(3):
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=60)

            try:
                result = response.json()
            except ValueError:
                result = None

            if isinstance(result, list):
                return result

            if isinstance(result, dict):
                last_error = result.get("error") or result.get("message") or json.dumps(result)
            else:
                last_error = response.text[:500] or f"HTTP {response.status_code}"

            if attempt < 2:
                time.sleep(5)

        raise RuntimeError(f"Hugging Face embeddings request failed: {last_error}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def resolve_chroma_persist_directory() -> str:
    env_path = os.environ.get("CHROMA_PERSIST_DIR")
    candidates: List[Path] = []

    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    script_dir = Path(__file__).resolve().parent
    candidates.append((script_dir / "my_vector_db").resolve())
    candidates.append((Path.cwd() / "my_vector_db").resolve())

    for candidate in candidates:
        if (candidate / "chroma.sqlite3").exists():
            return str(candidate)

    return str(candidates[0])


def truncate_text(value: Any, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text

    return f"{text[: limit - 3].rstrip()}..."


def build_tool_error(tool_name: str, query: str, error_type: str, message: str) -> ToolExecutionPayload:
    return ToolExecutionPayload(
        success=False,
        tool_name=tool_name,
        query=query,
        error=ToolErrorPayload(type=error_type, message=message),
    )


def validate_query(query: str, tool_name: str) -> Optional[ToolExecutionPayload]:
    normalized = (query or "").strip()
    if len(normalized) < 2 or len(normalized) > 300:
        return build_tool_error(
            tool_name=tool_name,
            query=normalized,
            error_type="validation_error",
            message="Query must be between 2 and 300 characters.",
        )

    return None


embedding_model = BulletproofHFEmbeddings()
db = Chroma(persist_directory=resolve_chroma_persist_directory(), embedding_function=embedding_model)


@mcp.tool()
def search_resume_database(query: str) -> str:
    validation_error = validate_query(query, "search_resume_database")
    if validation_error is not None:
        return validation_error.model_dump_json()

    try:
        docs = db.similarity_search(query, k=DEFAULT_RESUME_RESULTS)
        results = []
        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "content": truncate_text(doc.page_content),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                }
            )

        payload = ToolExecutionPayload(
            success=True,
            tool_name="search_resume_database",
            query=query,
            results=results,
        )
        return payload.model_dump_json()
    except Exception as exc:
        return build_tool_error(
            tool_name="search_resume_database",
            query=query,
            error_type="tool_execution_error",
            message=f"Resume database search failed: {exc}",
        ).model_dump_json()


@mcp.tool()
def search_live_web(query: str) -> str:
    validation_error = validate_query(query, "search_live_web")
    if validation_error is not None:
        return validation_error.model_dump_json()

    if tavily_client is None:
        return build_tool_error(
            tool_name="search_live_web",
            query=query,
            error_type="configuration_error",
            message="TAVILY_API_KEY is not configured.",
        ).model_dump_json()

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=DEFAULT_WEB_RESULTS,
        )
        results = [
            {
                "url": result.get("url", ""),
                "content": truncate_text(result.get("content", "")),
            }
            for result in response.get("results", [])
        ]
        payload = ToolExecutionPayload(
            success=True,
            tool_name="search_live_web",
            query=query,
            results=results,
        )
        return payload.model_dump_json()
    except Exception as exc:
        return build_tool_error(
            tool_name="search_live_web",
            query=query,
            error_type="tool_execution_error",
            message=f"Live web search failed: {exc}",
        ).model_dump_json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
