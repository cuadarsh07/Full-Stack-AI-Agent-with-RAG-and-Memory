"""
Advanced RAG Pipeline -- rag_pipeline.py
========================================
A fully self-contained drop-in module that upgrades the existing system with:
  1. Dynamic Indexing & Storage
  2. Advanced Retrieval (Pre, During, Post)
  3. Grounded Citation-based Generation
  4. Evaluation Hooks (Ragas / LangSmith)

No existing file is modified. Import this module in main.py.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("rag_pipeline")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment / Constants
# ---------------------------------------------------------------------------
PINECONE_API_KEY: Optional[str] = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "portfolio-rag-chunks")
PINECONE_CLOUD: str = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.environ.get("PINECONE_REGION", "us-east-1")
GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
LANGCHAIN_API_KEY: Optional[str] = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING: bool = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() in {"1", "true", "yes"}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

USER_DOC_NAMESPACE_PREFIX = "user-docs-"
RESUME_NAMESPACE = "portfolio-rag"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CANDIDATE_FETCH_MULTIPLIER = 4
MMR_LAMBDA = 0.6
TOP_K_DEFAULT = 5
MAX_COMPRESSION_CHARS = 1200

# ---------------------------------------------------------------------------
# Module-level singleton caches (thread-safe lazy init)
# ---------------------------------------------------------------------------
_embedding_model: Optional[Any] = None
_embedding_model_lock = threading.Lock()
_cross_encoder: Optional[Any] = None
_cross_encoder_lock = threading.Lock()
_pinecone_index: Optional[Any] = None
_pinecone_index_lock = threading.Lock()


def _get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model
        from sentence_transformers import SentenceTransformer
        LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_cross_encoder() -> Any:
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            LOGGER.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL_NAME)
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        except Exception as exc:
            LOGGER.warning("Cross-encoder unavailable (%s); reranking disabled.", exc)
            _cross_encoder = None
    return _cross_encoder


def _get_pinecone_index() -> Any:
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not configured.")
    with _pinecone_index_lock:
        if _pinecone_index is not None:
            return _pinecone_index
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        try:
            _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Pinecone index '{PINECONE_INDEX_NAME}': {exc}"
            ) from exc
        LOGGER.info("Pinecone index '%s' connected.", PINECONE_INDEX_NAME)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Unified LLM caller -- prefers Gemini, falls back to Groq."""
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types as genai_types
            client = genai.Client(api_key=GEMINI_API_KEY)
            config_kwargs: Dict[str, Any] = {"system_instruction": system_prompt}
            if json_mode:
                config_kwargs["response_mime_type"] = "application/json"
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            return (response.text or "").strip()
        except Exception as exc:
            LOGGER.warning("Gemini LLM call failed (%s); falling back to Groq.", exc)

    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            kwargs: Dict[str, Any] = {
                "model": GROQ_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            LOGGER.error("Groq LLM call failed: %s", exc)
            raise

    raise RuntimeError("No LLM available. Configure GEMINI_API_KEY or GROQ_API_KEY.")


# ===========================================================================
# 1. DYNAMIC INDEXING & STORAGE
# ===========================================================================

class RecursiveChunker:
    """
    Robust recursive character text splitter.
    Handles: paragraphs -> sentences -> words -> characters.
    Preserves semantic boundaries via overlap.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                separators=self.SEPARATORS,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                keep_separator=False,
            )
        except ImportError:
            self._splitter = None
            LOGGER.warning("langchain_text_splitters unavailable; using simple splitter.")

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if self._splitter is not None:
            chunks = self._splitter.split_text(text)
        else:
            chunks = self._simple_split(text)
        return [c.strip() for c in chunks if c.strip()]

    def _simple_split(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= self.chunk_size:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)
        return chunks


class DocumentIngestionService:
    """Extracts raw text from uploaded documents. Supports: .pdf, .txt, .md, .docx"""

    @staticmethod
    def extract_text(filename: str, file_bytes: bytes) -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return DocumentIngestionService._extract_pdf(file_bytes)
        elif ext in {".txt", ".md"}:
            return file_bytes.decode("utf-8", errors="replace")
        elif ext == ".docx":
            return DocumentIngestionService._extract_docx(file_bytes)
        else:
            return file_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return "\n\n".join(pages)

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            LOGGER.warning("python-docx not installed; returning raw bytes as text.")
            return file_bytes.decode("utf-8", errors="replace")


@dataclass
class IngestedDocument:
    doc_id: str
    filename: str
    namespace: str
    chunk_count: int
    char_count: int
    ingestion_time_ms: float


class PineconeIndexService:
    """
    Dynamically upserts document chunks into Pinecone.
    Each document gets its own namespace: user-docs-{doc_id}.
    Stores text payload inside vector metadata for retrieval.
    """

    BATCH_SIZE = 100

    def __init__(self) -> None:
        self._chunker = RecursiveChunker()

    def ingest(
        self,
        filename: str,
        file_bytes: bytes,
        doc_id: Optional[str] = None,
    ) -> IngestedDocument:
        started = time.perf_counter()
        doc_id = doc_id or _sha256(f"{filename}{len(file_bytes)}{uuid.uuid4().hex}")[:16]
        namespace = f"{USER_DOC_NAMESPACE_PREFIX}{doc_id}"

        LOGGER.info("Ingesting '%s' -> namespace '%s'", filename, namespace)

        raw_text = DocumentIngestionService.extract_text(filename, file_bytes)
        if not raw_text.strip():
            raise ValueError(f"No extractable text found in '{filename}'.")

        chunks = self._chunker.split(raw_text)
        LOGGER.info("'%s' produced %d chunks.", filename, len(chunks))

        embeddings = _embed(chunks)
        index = _get_pinecone_index()
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}-{i}"
            vectors.append({
                "id": chunk_id,
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk[:2000],
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "namespace": namespace,
                },
            })

        for batch_start in range(0, len(vectors), self.BATCH_SIZE):
            batch = vectors[batch_start: batch_start + self.BATCH_SIZE]
            index.upsert(vectors=batch, namespace=namespace)

        elapsed = (time.perf_counter() - started) * 1000
        LOGGER.info("Upserted %d vectors for '%s' in %.0f ms.", len(vectors), filename, elapsed)

        return IngestedDocument(
            doc_id=doc_id,
            filename=filename,
            namespace=namespace,
            chunk_count=len(chunks),
            char_count=len(raw_text),
            ingestion_time_ms=round(elapsed, 2),
        )

    @staticmethod
    def list_user_namespaces() -> List[Dict[str, Any]]:
        """Returns a list of all user-document namespaces in the Pinecone index."""
        try:
            index = _get_pinecone_index()
            stats = index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", {}) or {}
            return [
                {
                    "namespace": ns,
                    "doc_id": ns.replace(USER_DOC_NAMESPACE_PREFIX, "", 1),
                    "vector_count": (
                        info.vector_count
                        if hasattr(info, "vector_count")
                        else info.get("vector_count", 0)
                    ),
                }
                for ns, info in namespaces.items()
                if ns.startswith(USER_DOC_NAMESPACE_PREFIX)
            ]
        except Exception as exc:
            LOGGER.warning("Could not list Pinecone namespaces: %s", exc)
            return []


# ===========================================================================
# 2. ADVANCED RETRIEVAL -- PRE-RETRIEVAL
# ===========================================================================

class QueryRewriter:
    """Uses an LLM to clarify and expand the user query before retrieval."""

    SYSTEM_PROMPT = (
        "You are a search query optimization assistant. "
        "Rewrite the user question into a clear, specific, self-contained search query "
        "that captures the exact information need. "
        "Remove vague pronouns (this, that, it). "
        "Keep it under 120 characters. Output ONLY the rewritten query, no explanation."
    )

    def rewrite(self, query: str) -> str:
        if not query or len(query.strip()) < 5:
            return query
        try:
            rewritten = _call_llm(self.SYSTEM_PROMPT, query.strip())
            rewritten = re.sub(r"^[\"\'\`]|[\"\'\`]$", "", rewritten.strip())
            return rewritten[:200] if rewritten else query
        except Exception as exc:
            LOGGER.warning("Query rewriting failed (%s); using original.", exc)
            return query


class MultiQueryGenerator:
    """
    Generates N semantically diverse reformulations of the query.
    Increases recall by covering different aspects of the information need.
    """

    SYSTEM_PROMPT = (
        "You are a search diversity expert. Given a user query, generate {n} "
        "different phrasings that capture different aspects or angles of the same "
        "information need. Each phrasing should be distinct from the others. "
        "Output ONLY a JSON array of strings: [\"query1\", \"query2\", ...]"
    )

    def generate(self, query: str, n: int = 3) -> List[str]:
        if not query or not query.strip():
            return [query]
        try:
            prompt = self.SYSTEM_PROMPT.format(n=n)
            raw = _call_llm(prompt, query.strip(), json_mode=True)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if str(q).strip()][:n]
            elif isinstance(parsed, dict):
                queries = [
                    str(q).strip()
                    for q in (list(parsed.values())[0] or [])
                    if str(q).strip()
                ][:n]
            else:
                queries = []
            return queries if queries else [query]
        except Exception as exc:
            LOGGER.warning("Multi-query generation failed (%s); using original only.", exc)
            return [query]


class DomainRouter:
    """
    Routes the query to the correct Pinecone namespace.
    When doc_id is provided, routes to user-docs-{doc_id}.
    Otherwise falls back to resume namespace for backward compatibility.
    """

    RESUME_KEYWORDS = re.compile(
        r"\b(adarsh|resume|cv|profile|portfolio|candidate|internship|neeve)\b",
        re.IGNORECASE,
    )

    def route(self, query: str, doc_id: Optional[str] = None) -> str:
        if doc_id:
            return f"{USER_DOC_NAMESPACE_PREFIX}{doc_id}"
        if self.RESUME_KEYWORDS.search(query):
            return RESUME_NAMESPACE
        return RESUME_NAMESPACE


# ===========================================================================
# 2. ADVANCED RETRIEVAL -- DURING RETRIEVAL
# ===========================================================================

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    filename: str
    chunk_index: int
    doc_id: str
    namespace: str
    rerank_score: Optional[float] = None
    compressed_text: Optional[str] = None


class HybridRetriever:
    """
    Combines:
    1. Pinecone dense vector search (semantic similarity)
    2. BM25 sparse keyword search (exact term matching)
    3. Reciprocal Rank Fusion (RRF) to merge both rankings
    """

    RRF_K = 60  # standard RRF constant

    def retrieve(
        self,
        query: str,
        multi_queries: List[str],
        namespace: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> List[RetrievedChunk]:
        candidate_k = top_k * CANDIDATE_FETCH_MULTIPLIER
        all_queries = list(dict.fromkeys([query] + multi_queries))
        dense_hits: Dict[str, RetrievedChunk] = {}
        dense_ranks: Dict[str, Dict[str, int]] = {}

        for q_idx, q in enumerate(all_queries):
            try:
                hits = self._dense_search(q, namespace, candidate_k)
                q_key = f"q{q_idx}"
                dense_ranks[q_key] = {}
                for rank, hit in enumerate(hits):
                    dense_ranks[q_key][hit.chunk_id] = rank
                    if hit.chunk_id not in dense_hits:
                        dense_hits[hit.chunk_id] = hit
            except Exception as exc:
                LOGGER.warning("Dense search failed for query '%s': %s", q, exc)

        if not dense_hits:
            return []

        candidates = list(dense_hits.values())
        bm25_ranks = self._bm25_rank(query, candidates)

        rrf_scores: Dict[str, float] = {}
        for chunk_id in dense_hits:
            rrf_score = 0.0
            for q_key, ranks in dense_ranks.items():
                rank = ranks.get(chunk_id, len(candidates))
                rrf_score += 1.0 / (self.RRF_K + rank + 1)
            bm25_rank = bm25_ranks.get(chunk_id, len(candidates))
            rrf_score += 1.0 / (self.RRF_K + bm25_rank + 1)
            rrf_scores[chunk_id] = rrf_score

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        ranked = []
        for cid in sorted_ids[:candidate_k]:
            chunk = dense_hits[cid]
            chunk.score = rrf_scores[cid]
            ranked.append(chunk)
        return ranked[:candidate_k]

    def _dense_search(self, query: str, namespace: str, top_k: int) -> List[RetrievedChunk]:
        index = _get_pinecone_index()
        q_embedding = _embed([query])[0].tolist()
        result = index.query(
            vector=q_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        chunks: List[RetrievedChunk] = []
        for match in getattr(result, "matches", []) or []:
            meta = getattr(match, "metadata", {}) or {}
            text = meta.get("text") or meta.get("content") or ""
            if not text.strip():
                continue
            chunks.append(RetrievedChunk(
                chunk_id=getattr(match, "id", str(uuid.uuid4())),
                text=text,
                score=float(getattr(match, "score", 0.0)),
                filename=meta.get("filename", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                doc_id=meta.get("doc_id", ""),
                namespace=namespace,
            ))
        return chunks

    def _bm25_rank(self, query: str, candidates: List[RetrievedChunk]) -> Dict[str, int]:
        if not candidates:
            return {}
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [c.text.lower().split() for c in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            ranked_indices = np.argsort(scores)[::-1]
            return {candidates[idx].chunk_id: rank for rank, idx in enumerate(ranked_indices)}
        except Exception as exc:
            LOGGER.warning("BM25 ranking failed (%s); skipping sparse signal.", exc)
            return {}


class MMRSelector:
    """
    Maximal Marginal Relevance selection.
    Balances relevance to query with diversity among selected chunks.
    MMR(d) = argmax [ lambda * sim(d,q) - (1-lambda) * max sim(d, d_i) ]
    """

    def __init__(self, lambda_val: float = MMR_LAMBDA) -> None:
        self.lambda_val = lambda_val

    def select(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = TOP_K_DEFAULT,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        query_embedding = _embed([query])[0]
        doc_embeddings = _embed([c.text for c in candidates])
        relevance = np.array([
            _cosine_similarity(doc_embeddings[i], query_embedding)
            for i in range(len(candidates))
        ])

        selected_indices: List[int] = []
        remaining = list(range(len(candidates)))

        first = int(np.argmax(relevance))
        selected_indices.append(first)
        remaining.remove(first)

        while len(selected_indices) < top_k and remaining:
            mmr_scores = []
            for idx in remaining:
                rel = relevance[idx]
                max_sim = max(
                    _cosine_similarity(doc_embeddings[idx], doc_embeddings[sel])
                    for sel in selected_indices
                )
                mmr = self.lambda_val * rel - (1 - self.lambda_val) * max_sim
                mmr_scores.append((idx, mmr))
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]


class CrossEncoderReranker:
    """
    Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.
    Scores each (query, passage) pair jointly for precise relevance.
    Falls back gracefully if model is unavailable.
    """

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        model = _get_cross_encoder()
        if model is None:
            LOGGER.info("Cross-encoder unavailable; skipping reranking.")
            return chunks
        try:
            pairs = [(query, c.text[:512]) for c in chunks]
            scores = model.predict(pairs)
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
            reranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
            LOGGER.info("Cross-encoder reranked %d chunks.", len(reranked))
            return reranked
        except Exception as exc:
            LOGGER.warning("Cross-encoder reranking failed (%s); keeping original order.", exc)
            return chunks


# ===========================================================================
# 2. ADVANCED RETRIEVAL -- POST-RETRIEVAL
# ===========================================================================

class ContextualCompressor:
    """
    LLM-based contextual compression.
    For each retrieved chunk, extracts only the sentences relevant to the query.
    Reduces noise before passing context to the generator.
    """

    SYSTEM_PROMPT = (
        "You are a context distiller. Given a question and a passage, "
        "extract ONLY the sentences from the passage that are directly relevant "
        "to answering the question. "
        "Do NOT add, infer, or paraphrase -- copy verbatim. "
        "If no part of the passage is relevant, respond with exactly: IRRELEVANT"
    )

    def compress(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        compressed: List[RetrievedChunk] = []
        for chunk in chunks:
            try:
                user_prompt = (
                    f"Question: {query}\n\n"
                    f"Passage:\n{chunk.text[:MAX_COMPRESSION_CHARS]}"
                )
                result = _call_llm(self.SYSTEM_PROMPT, user_prompt)
                if result.strip().upper() == "IRRELEVANT" or not result.strip():
                    continue
                chunk.compressed_text = result.strip()
                compressed.append(chunk)
            except Exception as exc:
                LOGGER.warning(
                    "Compression failed for chunk %s (%s); keeping original.",
                    chunk.chunk_id, exc,
                )
                chunk.compressed_text = chunk.text
                compressed.append(chunk)
        LOGGER.info("Contextual compression: %d -> %d chunks.", len(chunks), len(compressed))
        return compressed


# ===========================================================================
# 3. AUGMENTATION & GENERATION
# ===========================================================================

@dataclass
class Citation:
    chunk_id: str
    filename: str
    chunk_index: int
    relevant_excerpt: str


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    rewritten_query: str
    multi_queries: List[str]
    retrieved_chunk_count: int
    final_chunk_count: int
    latency_ms: float
    evaluation_trace: Optional["RAGEvaluationHooks"] = None


CITATION_SYSTEM_PROMPT = """\
You are a precise, citation-driven document analyst. Your goal is to answer questions \
strictly from the provided context chunks.

STRICT RULES:
1. Answer ONLY using information explicitly present in the context chunks below.
2. If the context does not contain the answer, respond with:
   {"answer": "I don't have enough information in the provided documents to answer that.", "citations": []}
3. For every factual claim you make, append an inline citation marker like [Source 1], [Source 2], etc.
4. Do NOT speculate, infer beyond what is stated, or use any prior knowledge.
5. Keep the answer concise, clear, and well-structured (use markdown where appropriate).

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "answer": "Your grounded answer with [Source N] inline citations...",
  "citations": [
    {
      "source_id": 1,
      "filename": "example.pdf",
      "chunk_index": 0,
      "excerpt": "The exact sentence(s) from the chunk that support this claim."
    }
  ]
}"""


class CitationGenerator:
    """
    Grounded generation with inline citations.
    Enforces answer-only-from-context via strict prompt engineering.
    Returns structured JSON with answer and source attribution.
    """

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        use_compressed: bool = True,
    ) -> Tuple[str, List[Citation]]:
        if not chunks:
            return (
                "I don't have enough information in the provided documents to answer that.",
                [],
            )

        context_parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            text = (
                chunk.compressed_text
                if use_compressed and chunk.compressed_text
                else chunk.text
            )
            context_parts.append(
                f"[Source {i}] File: {chunk.filename} | Chunk #{chunk.chunk_index}\n{text}"
            )

        context_block = "\n\n---\n\n".join(context_parts)
        user_prompt = (
            f"Context Chunks:\n{context_block}\n\n"
            f"---\n\nQuestion: {query}"
        )

        try:
            raw = _call_llm(CITATION_SYSTEM_PROMPT, user_prompt, json_mode=True)
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except Exception:
                    parsed = {}
            else:
                parsed = {}
        except Exception as exc:
            LOGGER.error("Citation generation LLM call failed: %s", exc)
            return ("I could not generate an answer due to an internal error.", [])

        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            answer = "I don't have enough information in the provided documents to answer that."

        raw_citations = parsed.get("citations", []) or []
        citations: List[Citation] = []
        for cit in raw_citations:
            if not isinstance(cit, dict):
                continue
            source_id = int(cit.get("source_id", 1)) - 1
            if 0 <= source_id < len(chunks):
                ref_chunk = chunks[source_id]
                citations.append(Citation(
                    chunk_id=ref_chunk.chunk_id,
                    filename=cit.get("filename") or ref_chunk.filename,
                    chunk_index=int(cit.get("chunk_index", ref_chunk.chunk_index)),
                    relevant_excerpt=str(cit.get("excerpt", ""))[:500],
                ))

        return answer, citations


# ===========================================================================
# 4. EVALUATION HOOKS (Ragas / LangSmith)
# ===========================================================================

@dataclass
class RAGEvaluationHooks:
    """
    Full RAG trace captured at every pipeline step.

    Ready for Ragas:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        result = evaluate(hooks.to_ragas_dict(), metrics=[faithfulness, answer_relevancy])

    Ready for LangSmith:
        Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true in .env
        The pipeline functions will be auto-traced if langsmith is installed.
    """

    question: str
    rewritten_query: str
    multi_queries: List[str]
    namespace: str
    retrieved_chunks: List[Dict[str, Any]]
    mmr_selected_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    compressed_chunks: List[Dict[str, Any]]
    final_answer: str
    citations: List[Dict[str, Any]]
    latency_breakdown: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ragas_dict(self, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Convert to Ragas-compatible evaluation dict."""
        contexts = [
            c.get("compressed_text") or c.get("text", "")
            for c in self.compressed_chunks or self.reranked_chunks
        ]
        result: Dict[str, Any] = {
            "question": self.question,
            "answer": self.final_answer,
            "contexts": contexts,
        }
        if ground_truth:
            result["ground_truth"] = ground_truth
        return result

    def to_langsmith_metadata(self) -> Dict[str, Any]:
        """Metadata dict for LangSmith run tagging."""
        return {
            "question": self.question,
            "rewritten_query": self.rewritten_query,
            "multi_queries": self.multi_queries,
            "namespace": self.namespace,
            "retrieved_count": len(self.retrieved_chunks),
            "final_chunk_count": len(self.compressed_chunks),
            "answer_length": len(self.final_answer),
            **self.latency_breakdown,
        }


def _langsmith_trace(func):
    """Decorator stub for LangSmith tracing. Activates only when keys are set."""
    if not LANGCHAIN_API_KEY or not LANGCHAIN_TRACING:
        return func
    try:
        from langsmith import traceable
        return traceable(run_type="chain", name=func.__name__)(func)
    except ImportError:
        LOGGER.info("langsmith not installed; tracing skipped.")
        return func


# ===========================================================================
# 5. ORCHESTRATOR -- AdvancedRAGPipeline
# ===========================================================================

class AdvancedRAGPipeline:
    """
    Main orchestrator chaining all pipeline components.

    ingest() -> DocumentIngestionService + PineconeIndexService
    query()  -> QueryRewriter -> MultiQueryGenerator -> DomainRouter
             -> HybridRetriever -> MMRSelector -> CrossEncoderReranker
             -> ContextualCompressor -> CitationGenerator -> RAGResponse
    """

    def __init__(self) -> None:
        self._ingestion_service = PineconeIndexService()
        self._query_rewriter = QueryRewriter()
        self._multi_query_gen = MultiQueryGenerator()
        self._domain_router = DomainRouter()
        self._hybrid_retriever = HybridRetriever()
        self._mmr_selector = MMRSelector()
        self._reranker = CrossEncoderReranker()
        self._compressor = ContextualCompressor()
        self._citation_gen = CitationGenerator()

    def ingest_document(
        self,
        filename: str,
        file_bytes: bytes,
        doc_id: Optional[str] = None,
    ) -> IngestedDocument:
        return self._ingestion_service.ingest(filename, file_bytes, doc_id)

    @_langsmith_trace
    def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        top_k: int = TOP_K_DEFAULT,
        enable_compression: bool = True,
        enable_multi_query: bool = True,
    ) -> RAGResponse:
        pipeline_start = time.perf_counter()
        latency: Dict[str, float] = {}

        # -- Pre-Retrieval ----------------------------------------------------
        t0 = time.perf_counter()
        rewritten_query = self._query_rewriter.rewrite(question)
        latency["query_rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("Rewritten query: '%s'", rewritten_query)

        t0 = time.perf_counter()
        if enable_multi_query:
            multi_queries = self._multi_query_gen.generate(rewritten_query, n=3)
        else:
            multi_queries = [rewritten_query]
        latency["multi_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("Multi-queries: %s", multi_queries)

        t0 = time.perf_counter()
        namespace = self._domain_router.route(question, doc_id)
        latency["routing_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("Routing to namespace: '%s'", namespace)

        # -- Hybrid Retrieval -------------------------------------------------
        t0 = time.perf_counter()
        raw_candidates = self._hybrid_retriever.retrieve(
            query=rewritten_query,
            multi_queries=multi_queries,
            namespace=namespace,
            top_k=top_k,
        )
        latency["hybrid_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("Hybrid retrieval: %d candidates.", len(raw_candidates))

        if not raw_candidates:
            return RAGResponse(
                answer="I couldn't find any relevant information in the document for your question.",
                citations=[],
                rewritten_query=rewritten_query,
                multi_queries=multi_queries,
                retrieved_chunk_count=0,
                final_chunk_count=0,
                latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
            )

        # -- MMR Selection ----------------------------------------------------
        t0 = time.perf_counter()
        mmr_chunks = self._mmr_selector.select(
            query=rewritten_query,
            candidates=raw_candidates,
            top_k=top_k * 2,
        )
        latency["mmr_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("MMR selected %d chunks.", len(mmr_chunks))

        # -- Cross-Encoder Reranking ------------------------------------------
        t0 = time.perf_counter()
        reranked_chunks = self._reranker.rerank(rewritten_query, mmr_chunks)[:top_k]
        latency["reranking_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        LOGGER.info("After reranking: %d chunks.", len(reranked_chunks))

        # -- Contextual Compression -------------------------------------------
        if enable_compression:
            t0 = time.perf_counter()
            final_chunks = self._compressor.compress(rewritten_query, reranked_chunks)
            latency["compression_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        else:
            final_chunks = reranked_chunks
            latency["compression_ms"] = 0.0

        if not final_chunks:
            final_chunks = reranked_chunks
            LOGGER.info("Compression filtered all chunks; reverting to reranked set.")

        # -- Citation Generation ----------------------------------------------
        t0 = time.perf_counter()
        answer, citations = self._citation_gen.generate(
            query=question,
            chunks=final_chunks,
            use_compressed=enable_compression,
        )
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        latency["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 2)

        # -- Evaluation Trace -------------------------------------------------
        def _chunk_to_dict(c: RetrievedChunk) -> Dict[str, Any]:
            return {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "rerank_score": c.rerank_score,
                "compressed_text": c.compressed_text,
            }

        eval_hooks = RAGEvaluationHooks(
            question=question,
            rewritten_query=rewritten_query,
            multi_queries=multi_queries,
            namespace=namespace,
            retrieved_chunks=[_chunk_to_dict(c) for c in raw_candidates],
            mmr_selected_chunks=[_chunk_to_dict(c) for c in mmr_chunks],
            reranked_chunks=[_chunk_to_dict(c) for c in reranked_chunks],
            compressed_chunks=[_chunk_to_dict(c) for c in final_chunks],
            final_answer=answer,
            citations=[
                {
                    "chunk_id": cit.chunk_id,
                    "filename": cit.filename,
                    "chunk_index": cit.chunk_index,
                    "relevant_excerpt": cit.relevant_excerpt,
                }
                for cit in citations
            ],
            latency_breakdown=latency,
        )

        LOGGER.info(
            "RAG pipeline complete: %d citations, %.0f ms total.",
            len(citations),
            latency["total_ms"],
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            rewritten_query=rewritten_query,
            multi_queries=multi_queries,
            retrieved_chunk_count=len(raw_candidates),
            final_chunk_count=len(final_chunks),
            latency_ms=latency["total_ms"],
            evaluation_trace=eval_hooks,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (shared across FastAPI requests)
# ---------------------------------------------------------------------------
_pipeline_instance: Optional[AdvancedRAGPipeline] = None
_pipeline_lock = threading.Lock()


def get_rag_pipeline() -> AdvancedRAGPipeline:
    """Return the shared AdvancedRAGPipeline singleton (lazy initialised)."""
    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance
    with _pipeline_lock:
        if _pipeline_instance is not None:
            return _pipeline_instance
        _pipeline_instance = AdvancedRAGPipeline()
    return _pipeline_instance
