"""
Contextual chunking: split raw text into overlapping pieces
and prepend document-level context to each chunk for better
embedding quality (the "contextual retrieval" pattern).
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RawChunk = Tuple[str, dict]
IndexedChunk = Tuple[str, dict]  # (text, enriched_metadata)


def _build_context_prefix(meta: dict) -> str:
    """
    Build a short context header from metadata so the embedding
    captures *what document / section* a chunk belongs to.

    Example output:
      [Document: report.pdf | Type: pdf | Page: 3]
    """
    parts: List[str] = []
    source = meta.get("source") or meta.get("filename") or "unknown"
    parts.append(f"Document: {source}")

    file_type = meta.get("file_type")
    if file_type:
        parts.append(f"Type: {file_type}")

    page = meta.get("page_number")
    if page is not None:
        parts.append(f"Page: {page}")

    content_type = meta.get("content_type")  # paragraph / table from DOCX
    if content_type:
        parts.append(f"Section: {content_type}")

    return f"[{' | '.join(parts)}]\n"


def chunk_documents(
    raw_chunks: List[RawChunk],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[IndexedChunk]:
    """
    Split raw document chunks into embedding-sized pieces with
    contextual prefixes for improved retrieval quality.

    Returns a list of (text, metadata) where metadata includes
    a unique `chunk_id` combining source + chunk index.
    """
    try:
        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap or settings.chunk_overlap
    except Exception as e:
        logger.error("chunker_settings_error", error=str(e))
        size = chunk_size or 1000
        overlap = chunk_overlap or 200

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as e:
        logger.error("chunker_splitter_init_error", error=str(e))
        return []

    result: List[IndexedChunk] = []
    global_idx = 0

    try:
        for text, meta in raw_chunks:
            try:
                sub_texts = splitter.split_text(text)
            except Exception as e:
                logger.error(
                    "chunker_split_text_error",
                    error=str(e),
                    source=str(meta.get("source", "unknown")),
                )
                continue

            # Build context prefix once per parent chunk
            ctx_prefix = _build_context_prefix(meta)

            for local_idx, sub in enumerate(sub_texts):
                try:
                    enriched = dict(meta)
                    enriched["chunk_id"] = f"{meta.get('source', 'unknown')}::chunk_{global_idx}"
                    enriched["local_chunk_index"] = local_idx
                    enriched["global_chunk_index"] = global_idx

                    # Contextual chunking: prepend document context to the chunk text
                    # so the embedding model captures which document this came from.
                    contextual_text = ctx_prefix + sub

                    result.append((contextual_text, enriched))
                    global_idx += 1
                except Exception as e:
                    logger.error(
                        "chunker_enrich_append_error",
                        error=str(e),
                        source=str(meta.get("source", "unknown")),
                    )
                    continue
    except Exception as e:
        logger.error("chunker_loop_error", error=str(e))

    try:
        logger.info(
            "chunking_complete",
            raw_chunks=len(raw_chunks),
            final_chunks=len(result),
            chunk_size=size,
            contextual=True,
        )
    except Exception:
        pass

    return result