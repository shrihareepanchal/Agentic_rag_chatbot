"""
Load various file types into a uniform list of
(text: str, metadata: dict) tuples.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import List, Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)

RawChunk = Tuple[str, dict]  # (text, metadata)


def load_pdf(file_path: str) -> List[RawChunk]:
    """Extract text page-by-page from a PDF."""
    try:
        import pypdf
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")

    chunks: List[RawChunk] = []
    filename = Path(file_path).name

    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        continue
                    chunks.append(
                        (
                            text,
                            {
                                "source": filename,
                                "file_type": "pdf",
                                "page_number": page_num,
                                "total_pages": len(reader.pages),
                            },
                        )
                    )
                except Exception as e:
                    logger.error(
                        "pdf_page_extract_error",
                        filename=filename,
                        page_number=page_num,
                        error=str(e),
                    )
                    continue
    except Exception as e:
        logger.error("pdf_load_error", filename=filename, error=str(e))
        return []

    logger.info("pdf_loaded", filename=filename, pages=len(chunks))
    return chunks


def load_txt(file_path: str) -> List[RawChunk]:
    """Load a plain text file."""
    filename = Path(file_path).name
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except Exception as e:
        logger.error("txt_load_error", filename=filename, error=str(e))
        return []

    if not text:
        return []
    return [(text, {"source": filename, "file_type": "txt", "page_number": None})]


def load_docx(file_path: str) -> List[RawChunk]:
    """Load a DOCX file, one paragraph per raw chunk."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    filename = Path(file_path).name
    try:
        doc = Document(file_path)
    except Exception as e:
        logger.error("docx_open_error", filename=filename, error=str(e))
        return []

    chunks: List[RawChunk] = []

    para_idx = 0
    try:
        for para in doc.paragraphs:
            try:
                text = para.text.strip()
                if not text:
                    continue
                chunks.append(
                    (
                        text,
                        {
                            "source": filename,
                            "file_type": "docx",
                            "paragraph": para_idx,
                            "page_number": None,
                        },
                    )
                )
                para_idx += 1
            except Exception as e:
                logger.error(
                    "docx_paragraph_error",
                    filename=filename,
                    paragraph=para_idx,
                    error=str(e),
                )
                continue
    except Exception as e:
        logger.error("docx_iter_error", filename=filename, error=str(e))
        return []

    logger.info("docx_loaded", filename=filename, paragraphs=len(chunks))
    return chunks


def load_json(file_path: str) -> List[RawChunk]:
    """
    Flatten JSON into text chunks.
    - If the top-level is a list of objects: each object → one chunk.
    - If the top-level is an object: each key-value pair group → chunk.
    """
    filename = Path(file_path).name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("json_load_error", filename=filename, error=str(e))
        return []

    chunks: List[RawChunk] = []

    def obj_to_text(obj: dict, prefix: str = "") -> str:
        parts = []
        for k, v in obj.items():
            try:
                if isinstance(v, (dict, list)):
                    parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
                else:
                    parts.append(f"{k}: {v}")
            except Exception:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)

    try:
        if isinstance(data, list):
            for idx, item in enumerate(data):
                try:
                    text = obj_to_text(item) if isinstance(item, dict) else str(item)
                    chunks.append(
                        (
                            text,
                            {
                                "source": filename,
                                "file_type": "json",
                                "record_index": idx,
                                "page_number": None,
                            },
                        )
                    )
                except Exception as e:
                    logger.error("json_record_error", filename=filename, record_index=idx, error=str(e))
                    continue
        elif isinstance(data, dict):
            # Group top-level keys into chunks of ≤5 keys each
            items = list(data.items())
            for i in range(0, len(items), 5):
                try:
                    group = dict(items[i : i + 5])
                    text = obj_to_text(group)
                    chunks.append(
                        (
                            text,
                            {
                                "source": filename,
                                "file_type": "json",
                                "record_index": i // 5,
                                "page_number": None,
                            },
                        )
                    )
                except Exception as e:
                    logger.error("json_group_error", filename=filename, record_index=i // 5, error=str(e))
                    continue
    except Exception as e:
        logger.error("json_parse_loop_error", filename=filename, error=str(e))
        return []

    logger.info("json_loaded", filename=filename, chunks=len(chunks))
    return chunks


def load_csv(file_path: str) -> List[RawChunk]:
    """Load CSV: each row becomes a text chunk with column headers as context."""
    filename = Path(file_path).name
    chunks: List[RawChunk] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                try:
                    parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
                    text = " | ".join(parts)
                    if not text:
                        continue
                    chunks.append(
                        (
                            text,
                            {
                                "source": filename,
                                "file_type": "csv",
                                "row_index": row_idx,
                                "page_number": None,
                            },
                        )
                    )
                except Exception as e:
                    logger.error("csv_row_error", filename=filename, row_index=row_idx, error=str(e))
                    continue
    except Exception as e:
        logger.error("csv_load_error", filename=filename, error=str(e))
        return []

    logger.info("csv_loaded", filename=filename, rows=len(chunks))
    return chunks


def load_file(file_path: str) -> List[RawChunk]:
    """Dispatch to the correct loader based on file extension."""
    try:
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": load_pdf,
            ".txt": load_txt,
            ".md": load_txt,
            ".docx": load_docx,
            ".json": load_json,
            ".csv": load_csv,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader(file_path)
    except Exception as e:
        logger.error("load_file_error", file_path=str(file_path), error=str(e))
        return []


def supported_extensions() -> List[str]:
    return [".pdf", ".txt", ".md", ".docx", ".json", ".csv"]