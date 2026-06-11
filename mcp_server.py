import contextlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

DEFAULT_WEB_RESULTS = 3
DEFAULT_RESUME_RESULTS = 4
PINECONE_CANDIDATE_RESULTS = 12
MAX_TOOL_RESULT_CHARS = 2500
NAMESPACE = "portfolio-rag"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384
BASE_DIR = Path(__file__).resolve().parent
LOCAL_CHUNKS_PATH = BASE_DIR / "processed_chunks.json"
ENABLE_BACKGROUND_WARMUP = os.environ.get("ENABLE_MCP_BACKGROUND_WARMUP", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
USE_PINECONE_RESUME_SEARCH = os.environ.get("USE_PINECONE_RESUME_SEARCH", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SKILL_QUERY_TERMS = {
    "backend",
    "ci/cd",
    "cloud",
    "core",
    "devops",
    "framework",
    "frameworks",
    "interpersonal",
    "java",
    "language",
    "languages",
    "microservice",
    "microservices",
    "qualification",
    "qualifications",
    "qualified",
    "skill",
    "skills",
    "spring",
    "stack",
    "tech",
    "technical",
    "technologies",
    "technology",
    "tool",
    "tools",
}
EDUCATION_QUERY_TERMS = {
    "academic",
    "academics",
    "b.e",
    "be",
    "cgpa",
    "degree",
    "education",
    "educational",
    "marks",
    "qualification",
    "qualifications",
    "school",
    "university",
}
EXPERIENCE_QUERY_TERMS = {
    "company",
    "experience",
    "job",
    "neeve",
    "outlier",
    "position",
    "responsibilities",
    "role",
    "work",
    "worked",
}
PROJECT_QUERY_TERMS = {
    "achievement",
    "achievements",
    "application",
    "banking",
    "built",
    "project",
    "projects",
}
PROFILE_QUERY_TERMS = {
    "background",
    "bio",
    "contact",
    "email",
    "github",
    "introduction",
    "linkedin",
    "overview",
    "phone",
    "profile",
    "summary",
    "who",
}

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "portfolio-rag-chunks")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

mcp = FastMCP("Adarsh Portfolio Server")
logger = logging.getLogger("portfolio_mcp_server")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


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


def truncate_text(value: Any, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text

    return f"{text[: limit - 3].rstrip()}..."


def tokenize(value: str) -> set[str]:
    normalized = value.lower().replace("/", " ").replace("-", " ")
    return {
        token.strip(".,:;!?()[]{}\"'")
        for token in normalized.split()
        if token.strip(".,:;!?()[]{}\"'")
    }


def build_tool_error(tool_name: str, query: str, error_type: str, message: str) -> ToolExecutionPayload:
    return ToolExecutionPayload(
        success=False,
        tool_name=tool_name,
        query=query,
        error=ToolErrorPayload(type=error_type, message=message),
    )


def build_resume_failure_payload(query: str, exc: BaseException) -> ToolExecutionPayload:
    return build_tool_error(
        tool_name="search_resume_database",
        query=query,
        error_type="tool_execution_error",
        message=f"Resume database search failed: {exc}",
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


embedding_model: Optional[Any] = None
pinecone_index: Optional[Any] = None
tavily_client: Optional[Any] = None
local_resume_chunks: Optional[List[Dict[str, Any]]] = None
embedding_model_lock = threading.Lock()
pinecone_index_lock = threading.Lock()
tavily_client_lock = threading.Lock()
local_resume_chunks_lock = threading.Lock()


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def log_tool_timing(tool_name: str, timings: Dict[str, Any]) -> None:
    logger.info(json.dumps({"event": "mcp_tool_timing", "tool": tool_name, **timings}, default=str))


def log_tool_exception(tool_name: str, query: str, exc: BaseException, timings: Dict[str, Any]) -> None:
    logger.error(
        json.dumps(
            {
                "event": "mcp_tool_exception",
                "tool": tool_name,
                "query": query,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "timing": timings,
            },
            default=str,
        )
    )


def get_match_value(match: Any, key: str, default: Any = None) -> Any:
    if isinstance(match, dict):
        return match.get(key, default)

    return getattr(match, key, default)


@contextlib.contextmanager
def suppress_model_output():
    # MCP stdio uses stdout as its protocol stream. Redirecting sys.stdout or
    # sys.stderr is process-global and can race with FastMCP's own IO, so this
    # context deliberately does not swap descriptors.
    yield


def get_embedding_model() -> Any:
    global embedding_model

    if embedding_model is not None:
        return embedding_model

    with embedding_model_lock:
        if embedding_model is not None:
            return embedding_model

        from sentence_transformers import SentenceTransformer

        with suppress_model_output():
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

    return embedding_model


def get_pinecone_index() -> Any:
    global pinecone_index

    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")

    if pinecone_index is not None:
        return pinecone_index

    with pinecone_index_lock:
        if pinecone_index is not None:
            return pinecone_index

        from pinecone import Pinecone

        with suppress_model_output():
            pinecone_client = Pinecone(api_key=pinecone_api_key)
            try:
                pinecone_index = pinecone_client.index(pinecone_index_name)
            except AttributeError:
                pinecone_index = pinecone_client.Index(pinecone_index_name)

    return pinecone_index


def get_tavily_client() -> Any:
    global tavily_client

    if not tavily_api_key:
        return None

    if tavily_client is not None:
        return tavily_client

    with tavily_client_lock:
        if tavily_client is not None:
            return tavily_client

        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=tavily_api_key)

    return tavily_client


def get_local_resume_chunks() -> List[Dict[str, Any]]:
    global local_resume_chunks

    if local_resume_chunks is not None:
        return local_resume_chunks

    with local_resume_chunks_lock:
        if local_resume_chunks is not None:
            return local_resume_chunks

        if not LOCAL_CHUNKS_PATH.exists():
            local_resume_chunks = []
            return local_resume_chunks

        with LOCAL_CHUNKS_PATH.open("r", encoding="utf-8") as handle:
            loaded_chunks = json.load(handle)

        local_resume_chunks = loaded_chunks if isinstance(loaded_chunks, list) else []
        return local_resume_chunks


def decode_nested_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        return nested_metadata

    if isinstance(nested_metadata, str):
        try:
            parsed = json.loads(nested_metadata)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed

    return {}


def map_pinecone_match(match: Any, rank: int) -> Dict[str, Any]:
    metadata = get_match_value(match, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    nested_metadata = decode_nested_metadata(metadata)
    merged_metadata = {**nested_metadata, **metadata}
    merged_metadata.pop("metadata", None)

    content = (
        metadata.get("text")
        or metadata.get("content")
        or nested_metadata.get("text")
        or nested_metadata.get("content")
        or ""
    )

    return {
        "rank": rank,
        "content": truncate_text(content),
        "text": truncate_text(content),
        "score": get_match_value(match, "score"),
        "metadata": merged_metadata,
        "source": merged_metadata.get("source"),
        "page": merged_metadata.get("page") or merged_metadata.get("page_number"),
        "section_name": merged_metadata.get("section_name"),
    }


def section_boost(query_terms: set[str], section_name: Any, content: str) -> float:
    section = str(section_name or "").lower()
    boost = 0.0
    asks_for_profile = bool(query_terms & PROFILE_QUERY_TERMS)

    if section == "profile" and not asks_for_profile:
        boost -= 0.25

    if section == "profile" and asks_for_profile:
        boost += 0.12

    if query_terms & SKILL_QUERY_TERMS:
        if section == "skills":
            boost += 0.35
        elif section in {"experience", "projects"}:
            boost += 0.08

    if query_terms & EDUCATION_QUERY_TERMS:
        if section == "education":
            boost += 0.25
        elif section == "skills":
            boost += 0.12

    if query_terms & EXPERIENCE_QUERY_TERMS and section == "experience":
        boost += 0.25

    if query_terms & PROJECT_QUERY_TERMS and section == "projects":
        boost += 0.25

    content_terms = tokenize(content)
    if query_terms & content_terms:
        boost += min(0.12, 0.03 * len(query_terms & content_terms))

    return boost


def rerank_resume_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_terms = tokenize(query)

    def rank_key(result: Dict[str, Any]) -> float:
        score = result.get("score")
        base_score = float(score) if isinstance(score, (int, float)) else 0.0
        return base_score + section_boost(
            query_terms=query_terms,
            section_name=result.get("section_name"),
            content=result.get("content", ""),
        )

    reranked = sorted(results, key=rank_key, reverse=True)

    for rank, result in enumerate(reranked, start=1):
        result["rank"] = rank

    return reranked[:DEFAULT_RESUME_RESULTS]


def map_local_resume_chunk(chunk: Dict[str, Any], score: float, rank: int) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    content = chunk.get("text") or chunk.get("content") or ""
    return {
        "rank": rank,
        "content": truncate_text(content),
        "text": truncate_text(content),
        "score": score,
        "metadata": metadata,
        "source": metadata.get("source"),
        "page": metadata.get("page") or metadata.get("page_number"),
        "section_name": metadata.get("section_name"),
    }


def search_local_resume_chunks(query: str) -> List[Dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    scored_results: List[Dict[str, Any]] = []
    for chunk in get_local_resume_chunks():
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        content = str(chunk.get("text") or chunk.get("content") or "")
        content_terms = tokenize(content)
        overlap = query_terms & content_terms
        score = float(len(overlap))
        score += section_boost(query_terms, metadata.get("section_name"), content)

        if score > 0:
            scored_results.append(map_local_resume_chunk(chunk, score, len(scored_results) + 1))

    scored_results.sort(key=lambda result: float(result.get("score") or 0), reverse=True)
    return rerank_resume_results(query, scored_results[:PINECONE_CANDIDATE_RESULTS])


@mcp.tool()
def search_resume_database(query: str) -> str:
    total_started_at = time.perf_counter()
    timings: Dict[str, Any] = {
        "model_load_ms": 0.0,
        "embedding_ms": 0.0,
        "pinecone_query_ms": 0.0,
        "rerank_ms": 0.0,
        "local_search_ms": 0.0,
    }

    try:
        validation_error = validate_query(query, "search_resume_database")
        if validation_error is not None:
            timings["total_ms"] = elapsed_ms(total_started_at)
            log_tool_timing("search_resume_database", timings)
            return validation_error.model_dump_json()

        if not USE_PINECONE_RESUME_SEARCH:
            local_search_started_at = time.perf_counter()
            results = search_local_resume_chunks(query)
            timings["local_search_ms"] = elapsed_ms(local_search_started_at)
            timings["total_ms"] = elapsed_ms(total_started_at)
            log_tool_timing("search_resume_database", timings)
            return ToolExecutionPayload(
                success=True,
                tool_name="search_resume_database",
                query=query,
                results=results,
            ).model_dump_json()

        if not pinecone_api_key:
            timings["total_ms"] = elapsed_ms(total_started_at)
            log_tool_timing("search_resume_database", timings)
            return build_tool_error(
                tool_name="search_resume_database",
                query=query,
                error_type="configuration_error",
                message="PINECONE_API_KEY is not configured.",
            ).model_dump_json()

        model_load_started_at = time.perf_counter()
        model_was_loaded = embedding_model is not None
        model = get_embedding_model()
        if not model_was_loaded:
            timings["model_load_ms"] = elapsed_ms(model_load_started_at)

        embedding_started_at = time.perf_counter()
        with suppress_model_output():
            query_embedding = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        timings["embedding_ms"] = elapsed_ms(embedding_started_at)

        if len(query_embedding) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                "Expected "
                f"{EXPECTED_EMBEDDING_DIMENSION}-dimensional query embedding, "
                f"got {len(query_embedding)}."
            )

        pinecone_started_at = time.perf_counter()
        response = get_pinecone_index().query(
            vector=query_embedding,
            top_k=PINECONE_CANDIDATE_RESULTS,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        timings["pinecone_query_ms"] = elapsed_ms(pinecone_started_at)
        matches = get_match_value(response, "matches", []) or []
        candidate_results = [
            map_pinecone_match(match, rank)
            for rank, match in enumerate(matches, start=1)
        ]
        rerank_started_at = time.perf_counter()
        results = rerank_resume_results(query, candidate_results)
        timings["rerank_ms"] = elapsed_ms(rerank_started_at)

        payload = ToolExecutionPayload(
            success=True,
            tool_name="search_resume_database",
            query=query,
            results=results,
        )
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_timing("search_resume_database", timings)
        return payload.model_dump_json()
    except Exception as exc:
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_exception("search_resume_database", query, exc, timings)
        log_tool_timing("search_resume_database", timings)
        return build_resume_failure_payload(query, exc).model_dump_json()


@mcp.tool()
def search_live_web(query: str) -> str:
    total_started_at = time.perf_counter()
    timings: Dict[str, Any] = {
        "tavily_client_ms": 0.0,
        "tavily_search_ms": 0.0,
    }
    validation_error = validate_query(query, "search_live_web")
    if validation_error is not None:
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_timing("search_live_web", timings)
        return validation_error.model_dump_json()

    client_started_at = time.perf_counter()
    client = get_tavily_client()
    timings["tavily_client_ms"] = elapsed_ms(client_started_at)
    if client is None:
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_timing("search_live_web", timings)
        return build_tool_error(
            tool_name="search_live_web",
            query=query,
            error_type="configuration_error",
            message="TAVILY_API_KEY is not configured.",
        ).model_dump_json()

    try:
        search_started_at = time.perf_counter()
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=DEFAULT_WEB_RESULTS,
        )
        timings["tavily_search_ms"] = elapsed_ms(search_started_at)
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
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_timing("search_live_web", timings)
        return payload.model_dump_json()
    except Exception as exc:
        timings["total_ms"] = elapsed_ms(total_started_at)
        log_tool_timing("search_live_web", timings)
        return build_tool_error(
            tool_name="search_live_web",
            query=query,
            error_type="tool_execution_error",
            message=f"Live web search failed: {exc}",
        ).model_dump_json()


def warm_resume_dependencies() -> None:
    started_at = time.perf_counter()
    try:
        if USE_PINECONE_RESUME_SEARCH:
            get_embedding_model()
            get_pinecone_index()
        else:
            get_local_resume_chunks()
        log_tool_timing(
            "resume_dependency_warmup",
            {"success": True, "total_ms": elapsed_ms(started_at)},
        )
    except Exception as exc:
        log_tool_timing(
            "resume_dependency_warmup",
            {"success": False, "error": str(exc), "total_ms": elapsed_ms(started_at)},
        )


def start_background_warmup() -> None:
    if not ENABLE_BACKGROUND_WARMUP:
        log_tool_timing(
            "resume_dependency_warmup",
            {"success": True, "skipped": True, "reason": "disabled"},
        )
        return

    thread = threading.Thread(target=warm_resume_dependencies, name="resume-dependency-warmup", daemon=True)
    thread.start()


start_background_warmup()


if __name__ == "__main__":
    mcp.run(transport="stdio")
