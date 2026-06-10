import contextlib
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

load_dotenv()

DEFAULT_WEB_RESULTS = 3
DEFAULT_RESUME_RESULTS = 4
PINECONE_CANDIDATE_RESULTS = 12
MAX_TOOL_RESULT_CHARS = 2500
NAMESPACE = "portfolio-rag"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384
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


embedding_model: Optional[SentenceTransformer] = None
pinecone_index: Optional[Any] = None


def get_match_value(match: Any, key: str, default: Any = None) -> Any:
    if isinstance(match, dict):
        return match.get(key, default)

    return getattr(match, key, default)


@contextlib.contextmanager
def suppress_model_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def get_embedding_model() -> SentenceTransformer:
    global embedding_model

    if embedding_model is None:
        with suppress_model_output():
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return embedding_model


def get_pinecone_index() -> Any:
    global pinecone_index

    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")

    if pinecone_index is None:
        with suppress_model_output():
            pinecone_client = Pinecone(api_key=pinecone_api_key)
            try:
                pinecone_index = pinecone_client.index(pinecone_index_name)
            except AttributeError:
                pinecone_index = pinecone_client.Index(pinecone_index_name)

    return pinecone_index


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


@mcp.tool()
def search_resume_database(query: str) -> str:
    validation_error = validate_query(query, "search_resume_database")
    if validation_error is not None:
        return validation_error.model_dump_json()

    if not pinecone_api_key:
        return build_tool_error(
            tool_name="search_resume_database",
            query=query,
            error_type="configuration_error",
            message="PINECONE_API_KEY is not configured.",
        ).model_dump_json()

    try:
        with suppress_model_output():
            query_embedding = get_embedding_model().encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

        if len(query_embedding) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                "Expected "
                f"{EXPECTED_EMBEDDING_DIMENSION}-dimensional query embedding, "
                f"got {len(query_embedding)}."
            )

        response = get_pinecone_index().query(
            vector=query_embedding,
            top_k=PINECONE_CANDIDATE_RESULTS,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        matches = get_match_value(response, "matches", []) or []
        candidate_results = [
            map_pinecone_match(match, rank)
            for rank, match in enumerate(matches, start=1)
        ]
        results = rerank_resume_results(query, candidate_results)

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
