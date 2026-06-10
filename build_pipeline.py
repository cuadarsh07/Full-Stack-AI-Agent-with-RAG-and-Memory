import argparse
import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

try:
    from langchain_text_splitters import SemanticChunker
except ImportError:  # pragma: no cover - depends on installed LangChain extras.
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:  # pragma: no cover
        SemanticChunker = None


LOGGER = logging.getLogger("adarsh_ai.build_pipeline")

SOURCE_PDF = Path("Adarsh_Kumar_Resume.pdf")
OUTPUT_JSON = Path("processed_chunks.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_SECTION_NAME = "Profile"
MIN_CHILD_CHARS = 40
MAX_CHILD_CHARS = 900

KNOWN_SECTION_ALIASES = {
    "summary": "Summary",
    "profile": "Profile",
    "objective": "Profile",
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "employment": "Experience",
    "projects": "Projects",
    "project": "Projects",
    "skills": "Skills",
    "technical skills": "Skills",
    "education": "Education",
    "certifications": "Certifications",
    "certification": "Certifications",
    "achievements": "Achievements",
    "awards": "Achievements",
    "leadership": "Leadership",
    "contact": "Contact",
}

BULLET_PATTERN = re.compile(r"^\s*(?:[-*•‣▪]|\d+[.)])\s+")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MULTISPACE_PATTERN = re.compile(r"[ \t]+")
LEADING_WORD_SENTENCE_PATTERN = re.compile(r"^([a-z]+[.!?])\s+(.*)$")
PDF_ARTIFACT_REPLACEMENTS = {
    "â€“": "-",
    "â€”": "-",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€¦": "...",
    "â™‚phone": "phone",
    "âŒ¢pe": "",
}


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str
    section_name: str
    source: str
    content_hash: str
    chunk_index: int = Field(ge=0)


class VectorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ResumeLine:
    text: str
    page_number: int


@dataclass(frozen=True)
class ParentSection:
    section_name: str
    text: str
    parent_id: str
    start_page: int
    end_page: int


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def normalize_text(value: str) -> str:
    text = value.replace("\x00", " ")
    for broken, replacement in PDF_ARTIFACT_REPLACEMENTS.items():
        text = text.replace(broken, replacement)

    text = re.sub(r"/envel.*?pe(?=[A-Za-z0-9._%+-]+@)", "", text)
    text = text.replace("/github", "")
    text = text.replace("/linkedin", "")
    text = re.sub(r"/([A-Za-z]{3,})(?=\1)", "", text)
    text = text.replace("\u2022", "-")
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_pdf_lines(pdf_path: Path) -> List[ResumeLine]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    LOGGER.info("Extracting text from %s", pdf_path)
    reader = PdfReader(str(pdf_path))
    lines: List[ResumeLine] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            clean_line = normalize_text(raw_line)
            if clean_line:
                lines.append(ResumeLine(text=clean_line, page_number=page_index))

    lines = repair_pdf_line_wraps(lines)

    if not lines:
        raise ValueError(f"No extractable text found in {pdf_path}")

    LOGGER.info("Extracted %s non-empty text lines from %s page(s)", len(lines), len(reader.pages))
    return lines


def repair_pdf_line_wraps(lines: List[ResumeLine]) -> List[ResumeLine]:
    repaired: List[ResumeLine] = []
    index = 0

    while index < len(lines):
        current = lines[index]
        current_text = current.text

        if current_text.endswith("-") and index + 1 < len(lines):
            next_line = lines[index + 1]
            next_text = next_line.text.lstrip()
            leading_sentence = LEADING_WORD_SENTENCE_PATTERN.match(next_text)

            if leading_sentence:
                current_text = f"{current_text[:-1]}{leading_sentence.group(1)}"
                repaired.append(ResumeLine(text=normalize_text(current_text), page_number=current.page_number))

                remainder = normalize_text(leading_sentence.group(2))
                if remainder:
                    repaired.append(ResumeLine(text=remainder, page_number=next_line.page_number))
                index += 2
                continue

            if next_text and next_text[0].islower():
                current_text = f"{current_text[:-1]}{next_text}"
                repaired.append(ResumeLine(text=normalize_text(current_text), page_number=current.page_number))
                index += 2
                continue

        repaired.append(current)
        index += 1

    return repaired


def canonical_section_name(line: str) -> Optional[str]:
    candidate = normalize_text(line).strip(":").lower()
    candidate = re.sub(r"[^a-z\s]", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    if candidate in KNOWN_SECTION_ALIASES:
        return KNOWN_SECTION_ALIASES[candidate]

    if len(candidate.split()) <= 3 and candidate in KNOWN_SECTION_ALIASES.values():
        return candidate.title()

    return None


def looks_like_section_header(line: str) -> Optional[str]:
    explicit = canonical_section_name(line)
    if explicit:
        return explicit

    stripped = normalize_text(line).strip()
    if BULLET_PATTERN.match(stripped):
        return None

    words = stripped.split()
    alpha_chars = [char for char in stripped if char.isalpha()]
    if not alpha_chars or len(words) > 5:
        return None

    uppercase_ratio = sum(char.isupper() for char in alpha_chars) / len(alpha_chars)
    if uppercase_ratio >= 0.75:
        return stripped.title()

    return None


def looks_like_role_or_project_heading(line: str) -> bool:
    stripped = normalize_text(line)
    if BULLET_PATTERN.match(stripped):
        return False

    has_year = bool(re.search(r"\b20\d{2}\b", stripped))
    has_separator = any(separator in stripped for separator in (" - ", " – ", " — ", "|"))
    has_role_word = bool(
        re.search(
            r"\b(?:engineer|trainee|associate|developer|application|project|intern|remote|basis)\b",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    return has_year and (has_separator or has_role_word)


def build_parent_sections(lines: Iterable[ResumeLine]) -> List[ParentSection]:
    sections: List[ParentSection] = []
    current_name = DEFAULT_SECTION_NAME
    current_lines: List[ResumeLine] = []

    def flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return

        section_text = normalize_text("\n".join(line.text for line in current_lines))
        if not section_text:
            current_lines = []
            return

        parent_seed = f"{current_name}\n{section_text}"
        sections.append(
            ParentSection(
                section_name=current_name,
                text=section_text,
                parent_id=sha256_text(parent_seed),
                start_page=current_lines[0].page_number,
                end_page=current_lines[-1].page_number,
            )
        )
        current_lines = []

    for line in lines:
        section_name = looks_like_section_header(line.text)
        if section_name and current_lines:
            flush()
            current_name = section_name
            continue

        if section_name and not current_lines:
            current_name = section_name
            continue

        current_lines.append(line)

    flush()

    if not sections:
        raise ValueError("Could not build parent sections from extracted text.")

    LOGGER.info("Detected %s parent section(s)", len(sections))
    return sections


def split_oversized_text(text: str, max_chars: int = MAX_CHILD_CHARS) -> List[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(normalize_text(" ".join(current)))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += sentence_len + 1

    if current:
        chunks.append(normalize_text(" ".join(current)))

    return chunks


def split_by_bullets_and_sentences(text: str) -> List[str]:
    atomic_units: List[str] = []
    paragraph_buffer: List[str] = []
    bullet_buffer: List[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        paragraph = normalize_text(" ".join(paragraph_buffer))
        paragraph_buffer.clear()
        if paragraph:
            atomic_units.extend(split_oversized_text(paragraph))

    def flush_bullet() -> None:
        if not bullet_buffer:
            return
        bullet = normalize_text(" ".join(bullet_buffer))
        bullet_buffer.clear()
        if bullet:
            atomic_units.extend(split_oversized_text(bullet))

    for line in text.splitlines():
        clean_line = normalize_text(line)
        if not clean_line:
            flush_bullet()
            flush_paragraph()
            continue

        if BULLET_PATTERN.match(clean_line):
            flush_bullet()
            flush_paragraph()
            bullet_buffer.append(clean_line)
            continue

        if (
            bullet_buffer
            and not looks_like_section_header(clean_line)
            and not looks_like_role_or_project_heading(clean_line)
        ):
            bullet_buffer.append(clean_line)
            continue

        flush_bullet()
        paragraph_buffer.append(clean_line)

    flush_bullet()
    flush_paragraph()

    merged: List[str] = []
    current = ""
    for unit in atomic_units:
        if not current:
            current = unit
            continue

        if len(current) < MIN_CHILD_CHARS and len(current) + len(unit) + 1 <= MAX_CHILD_CHARS:
            current = normalize_text(f"{current} {unit}")
        else:
            merged.append(current)
            current = unit

    if current:
        merged.append(current)

    return [chunk for chunk in merged if chunk]


def semantic_chunk_section(section: ParentSection, embeddings: HuggingFaceEmbeddings) -> List[str]:
    fallback_chunks = split_by_bullets_and_sentences(section.text)

    if SemanticChunker is None:
        LOGGER.warning("SemanticChunker is unavailable; using bullet/sentence fallback.")
        return fallback_chunks

    try:
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        semantic_chunks = [
            normalize_text(chunk)
            for chunk in splitter.split_text(section.text)
            if normalize_text(chunk)
        ]
    except TypeError:
        splitter = SemanticChunker(embeddings)
        semantic_chunks = [
            normalize_text(chunk)
            for chunk in splitter.split_text(section.text)
            if normalize_text(chunk)
        ]
    except Exception as exc:
        LOGGER.warning(
            "Semantic chunking failed for section %s; using fallback. Error: %s",
            section.section_name,
            exc,
        )
        return fallback_chunks

    refined_chunks: List[str] = []
    for chunk in semantic_chunks:
        if len(chunk) > MAX_CHILD_CHARS:
            refined_chunks.extend(split_by_bullets_and_sentences(chunk))
        else:
            refined_chunks.append(chunk)

    return refined_chunks or fallback_chunks


def build_vector_payloads(sections: List[ParentSection]) -> List[VectorPayload]:
    LOGGER.info("Initializing local embedding model: %s", EMBEDDING_MODEL_NAME)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    payloads: List[VectorPayload] = []
    seen_hashes = set()

    for section in sections:
        child_chunks = semantic_chunk_section(section, embeddings)
        LOGGER.info(
            "Section '%s' produced %s child chunk(s)",
            section.section_name,
            len(child_chunks),
        )

        for chunk in child_chunks:
            clean_chunk = normalize_text(chunk)
            if not clean_chunk:
                continue

            content_hash = sha256_text(clean_chunk)
            if content_hash in seen_hashes:
                LOGGER.debug("Skipping duplicate chunk hash: %s", content_hash)
                continue

            chunk_index = len(payloads)
            payloads.append(
                VectorPayload(
                    id=content_hash,
                    text=clean_chunk,
                    metadata=ChunkMetadata(
                        parent_id=section.parent_id,
                        section_name=section.section_name,
                        source=SOURCE_PDF.name,
                        content_hash=content_hash,
                        chunk_index=chunk_index,
                    ),
                )
            )
            seen_hashes.add(content_hash)

    if not payloads:
        raise ValueError("Pipeline produced zero vector payloads.")

    LOGGER.info("Built %s unique vector payload(s)", len(payloads))
    return payloads


async def run_pipeline(pdf_path: Path = SOURCE_PDF, output_path: Path = OUTPUT_JSON) -> List[VectorPayload]:
    lines = await asyncio.to_thread(extract_pdf_lines, pdf_path)
    sections = await asyncio.to_thread(build_parent_sections, lines)
    payloads = await asyncio.to_thread(build_vector_payloads, sections)

    serialized_payloads = [payload.model_dump() for payload in payloads]
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(serialized_payloads, indent=2, ensure_ascii=False),
        "utf-8",
    )

    LOGGER.info("Wrote processed payloads to %s", output_path)
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured resume chunks for Adarsh AI.")
    parser.add_argument("--pdf", type=Path, default=SOURCE_PDF, help="Path to the source resume PDF.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Path for the output JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    asyncio.run(run_pipeline(pdf_path=args.pdf, output_path=args.output))


if __name__ == "__main__":
    main()
