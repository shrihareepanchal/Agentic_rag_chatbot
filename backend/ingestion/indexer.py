"""
High-level ingestion pipeline:
  file_path → load → chunk → embed → store in ChromaDB
  For CSV/JSON: also load into SQLite for SQL queries
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List

from backend.config import get_settings
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.document_loader import load_file
from backend.ingestion.embedder import get_embedder
from backend.retrieval.vector_store import get_vector_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _load_into_sql(file_path: str, filename: str) -> int | None:
    """
    If the file is CSV or JSON, also load it into SQLite for SQL queries.
    Returns the number of rows loaded, or None if not applicable.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in (".csv", ".json"):
        return None

    try:
        import pandas as pd
        from backend.retrieval.sql_store import get_sql_store

        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        else:
            # JSON: try reading as records array or line-delimited
            try:
                df = pd.read_json(file_path, lines=False)
            except ValueError:
                df = pd.read_json(file_path, lines=True)

        if df.empty:
            logger.warning("sql_load_empty_dataframe", filename=filename)
            return None

        # Sanitise table name from filename
        import re
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", Path(filename).stem).strip("_").lower()
        if not table_name:
            table_name = "uploaded_data"

        sql_store = get_sql_store()
        row_count = sql_store.create_table_from_dataframe(df, table_name, if_exists="replace")
        logger.info("sql_table_loaded", filename=filename, table=table_name, rows=row_count)
        return row_count

    except Exception as e:
        logger.warning("sql_load_failed", filename=filename, error=str(e))
        return None


def ingest_file(file_path: str) -> dict:
    """
    Full ingestion pipeline for a single file.

    Returns:
        {source_id, filename, file_type, chunks_indexed}
    """
    settings = get_settings()

    try:
        file_path = str(Path(file_path).resolve())
        filename = Path(file_path).name
        source_id = _make_source_id(filename)
    except Exception as e:
        logger.error("ingestion_path_error", error=str(e), file_path=str(file_path))
        raise

    logger.info("ingestion_start", filename=filename, source_id=source_id)

    # 0. De-duplicate: remove existing chunks for this source before re-ingesting
    try:
        vs = get_vector_store()
        deleted = vs.delete_by_source(filename)
        if deleted > 0:
            logger.info("dedup_deleted_old_chunks", filename=filename, deleted=deleted)
    except Exception as e:
        logger.warning("dedup_cleanup_failed", filename=filename, error=str(e))

    # 1. Load
    try:
        raw_chunks = load_file(file_path)
    except Exception as e:
        logger.error("ingestion_load_error", filename=filename, error=str(e))
        raise

    if not raw_chunks:
        raise ValueError(f"No text extracted from {filename}")

    # 2. Chunk
    try:
        indexed_chunks = chunk_documents(raw_chunks)
    except Exception as e:
        logger.error("ingestion_chunk_error", filename=filename, error=str(e))
        raise

    # 3. Embed
    try:
        embedder = get_embedder()
        texts = [t for t, _ in indexed_chunks]
        embeddings = embedder.embed_documents(texts)
    except Exception as e:
        logger.error("ingestion_embedding_error", filename=filename, error=str(e))
        raise

    # 4. Store in vector DB
    try:
        vs = get_vector_store()
        ids = [meta["chunk_id"] for _, meta in indexed_chunks]
        metadatas = [meta for _, meta in indexed_chunks]

        vs.add(
            texts=texts,              # ✅ changed from documents=texts
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error("ingestion_vector_store_error", filename=filename, error=str(e))
        raise

    file_type = Path(file_path).suffix.lstrip(".")

    # 5. For CSV/JSON: also load into SQL for aggregation queries
    sql_rows = _load_into_sql(file_path, filename)

    logger.info(
        "ingestion_complete",
        filename=filename,
        chunks=len(indexed_chunks),
        sql_rows=sql_rows,
    )
    return {
        "source_id": source_id,
        "filename": filename,
        "file_type": file_type,
        "chunks_indexed": len(indexed_chunks),
        "sql_rows": sql_rows,
    }


def _make_source_id(filename: str) -> str:
    try:
        return hashlib.sha1(filename.encode()).hexdigest()[:12]
    except Exception as e:
        logger.error("source_id_error", filename=str(filename), error=str(e))
        # keep deterministic-ish fallback without changing interface
        return "unknown_source"