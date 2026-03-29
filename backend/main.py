"""
FastAPI application entry point.

Endpoints:
  POST /chat            – Send a message, receive agentic response
  POST /upload          – Upload a file for ingestion
  POST /ingest          – Download & ingest a file from URL
  POST /scrape          – Scrape a web page & ingest its text
  GET  /sources         – List indexed sources
  DELETE /sources       – Clear all knowledge base data
  DELETE /sources/{fn}  – Remove a single document from the index
  GET  /sessions/{id}   – Retrieve conversation history
  DELETE /sessions/{id} – Clear a session
  GET  /sessions        – List all active sessions
  GET  /health          – Health check
  GET  /               – Serve chat UI (frontend/index.html)
"""
from __future__ import annotations

import sys

# Enforce Python 3.11
if sys.version_info < (3, 11):
    raise RuntimeError(f"Python 3.11+ required. Current version: {sys.version}")

import os
from contextlib import asynccontextmanager
from pathlib import Path

from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agents.graph import run_agent
from backend.config import get_settings
from backend.ingestion.indexer import ingest_file
from backend.memory.session_store import get_session_store
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    IngestURLRequest,
    MultiUploadResponse,
    ScrapeRequest,
    SessionHistory,
    SourceInfo,
    UploadResponse,
)
from backend.retrieval.sql_store import get_sql_store
from backend.retrieval.vector_store import get_vector_store
from backend.utils.logger import configure_logging, get_logger


# ── Rate Limiting ──────────────────────────────────────────────────────────────

from collections import defaultdict
import time
import ipaddress

# Simple in-memory rate limiter (no extra dependency needed)
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 30   # max requests per window
_RATE_WINDOW = 60  # seconds


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if rate limit exceeded."""
    now = time.time()
    timestamps = _rate_store[client_ip]
    # Clean old entries
    _rate_store[client_ip] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return True
    _rate_store[client_ip].append(now)
    return False


# ── SSRF Protection ───────────────────────────────────────────────────────────

_BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("10.0.0.0/8"),         # private
    ipaddress.ip_network("172.16.0.0/12"),      # private
    ipaddress.ip_network("192.168.0.0/16"),     # private
    ipaddress.ip_network("169.254.0.0/16"),     # link-local
    ipaddress.ip_network("0.0.0.0/8"),          # "this" network
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 private
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]


def _validate_url_safe(url: str) -> None:
    """
    SSRF protection: resolve the URL hostname and block internal/private IPs.
    Raises HTTPException if the URL targets an internal network.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname found.")

    # Block common internal hostnames
    blocked_hosts = {"localhost", "0.0.0.0", "127.0.0.1", "metadata.google.internal", "169.254.169.254"}
    if hostname.lower() in blocked_hosts:
        raise HTTPException(status_code=403, detail="Access to internal hosts is not allowed.")

    try:
        resolved = socket.getaddrinfo(hostname, None)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            for network in _BLOCKED_IP_RANGES:
                if ip in network:
                    raise HTTPException(
                        status_code=403,
                        detail="Access to internal/private IP addresses is not allowed.",
                    )
    except socket.gaierror:
        raise HTTPException(status_code=400, detail=f"Cannot resolve hostname: {hostname}")
    except HTTPException:
        raise
    except Exception:
        pass  # allow if resolution fails for non-security reasons

# ── App Lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("startup")
    log.info(
        "startup",
        mock_mode=settings.mock_mode,
        pii_redaction=settings.enable_pii_redaction,
        injection_guard=settings.enable_injection_guard,
    )
    _ = get_vector_store()
    _ = get_sql_store()
    yield
    log.info("shutdown")


# ── App Instance ──────────────────────────────────────────────────────────────

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("main")

app = FastAPI(
    title="Agentic RAG Chatbot",
    description="Production-style RAG with LangGraph agents, ChromaDB, and SQLite",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate Limit Middleware ─────────────────────────────────────────────────────


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to non-static endpoints."""
    if request.url.path.startswith("/static") or request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Max {_RATE_LIMIT} requests per {_RATE_WINDOW}s."},
        )
    return await call_next(request)

# ── Static Files ──────────────────────────────────────────────────────────────

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sanitise_filename(name: str) -> str:
    name = Path(name).name
    import re

    name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    return name[:128]


def _chroma_count(vs) -> int:
    try:
        store = getattr(vs, "_store", vs)
        col = getattr(store, "_collection", None)
        return int(col.count()) if col is not None else 0
    except Exception:
        return 0


def _chroma_sources(vs) -> list[dict]:
    """
    Build a per-source summary from Chroma metadatas.
    Expects metadata to include: source (filename), file_type, chunk_id/page optional.
    """
    store = getattr(vs, "_store", vs)
    col = getattr(store, "_collection", None)
    if col is None:
        return []

    data = col.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    stats: dict[str, dict] = {}
    for m in metadatas:
        if not isinstance(m, dict):
            continue
        fname = m.get("source") or m.get("filename") or m.get("file_path")
        if not fname:
            continue
        entry = stats.setdefault(
            fname,
            {
                "filename": fname,
                "file_type": m.get("file_type") or Path(str(fname)).suffix.lstrip(".") or "unknown",
                "chunk_count": 0,
            },
        )
        entry["chunk_count"] += 1

    return list(stats.values())


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Agentic RAG API", "docs": "/docs"})


@app.get("/health")
async def health_check():
    vs = get_vector_store()
    sql = get_sql_store()
    return {
        "status": "healthy",
        "mock_mode": settings.mock_mode,
        "vector_chunks_indexed": _chroma_count(vs),
        "sql_tables": sql.list_tables(),
        "sessions_active": len(get_session_store().list_sessions()),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Set a request ID for tracing this entire request through logs
    from backend.utils.logger import set_request_id
    req_id = set_request_id()
    logger.info("chat_request", session=request.session_id, message_len=len(request.message), request_id=req_id)

    session_store = get_session_store()
    history = session_store.get_messages(request.session_id)

    response = await run_agent(
        message=request.message,
        session_id=request.session_id,
        history=history,
        active_sources=request.active_sources,
        openai_api_key=request.openai_api_key,
    )

    session_store.add_user_message(request.session_id, request.message)
    session_store.add_ai_message(request.session_id, response.answer)

    logger.info(
        "chat_response",
        session=request.session_id,
        confidence=response.confidence,
        citations=len(response.citations),
        request_id=req_id,
    )
    return response


# ── File Upload ────────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file directly for ingestion into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    safe_name = _sanitise_filename(file.filename)
    from backend.ingestion.document_loader import supported_extensions
    ext = Path(safe_name).suffix.lower()
    if ext not in supported_extensions():
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(supported_extensions())}",
        )

    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb} MB")

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_file(save_path)
    except ValueError as e:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        os.remove(save_path)
        logger.error("upload_ingestion_error", filename=safe_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # Invalidate cached source names so agent picks up new file immediately
    try:
        from backend.agents.graph import invalidate_source_cache
        invalidate_source_cache()
    except Exception:
        pass

    return UploadResponse(
        status="success",
        filename=safe_name,
        source_id=result["source_id"],
        chunks_indexed=result["chunks_indexed"],
        file_type=result["file_type"],
        message=f"Successfully indexed {result['chunks_indexed']} chunks from {safe_name}",
    )


# ── Batch File Upload ─────────────────────────────────────────────────────


@app.post("/upload/batch", response_model=MultiUploadResponse)
async def upload_files_batch(files: List[UploadFile] = File(...)):
    """Upload multiple files at once for ingestion into the knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    from backend.ingestion.document_loader import supported_extensions

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    os.makedirs(settings.upload_dir, exist_ok=True)

    results: list[UploadResponse] = []
    errors: list[dict[str, str]] = []

    for file in files:
        if not file.filename:
            errors.append({"filename": "unknown", "error": "No filename provided"})
            continue

        safe_name = _sanitise_filename(file.filename)
        ext = Path(safe_name).suffix.lower()

        if ext not in supported_extensions():
            errors.append({
                "filename": safe_name,
                "error": f"Unsupported file type '{ext}'. Supported: {', '.join(supported_extensions())}",
            })
            continue

        content = await file.read()
        if len(content) > max_bytes:
            errors.append({
                "filename": safe_name,
                "error": f"File too large. Max: {settings.max_file_size_mb} MB",
            })
            continue

        save_path = os.path.join(settings.upload_dir, safe_name)
        with open(save_path, "wb") as f:
            f.write(content)

        try:
            result = ingest_file(save_path)
            results.append(UploadResponse(
                status="success",
                filename=safe_name,
                source_id=result["source_id"],
                chunks_indexed=result["chunks_indexed"],
                file_type=result["file_type"],
                message=f"Successfully indexed {result['chunks_indexed']} chunks from {safe_name}",
            ))
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            errors.append({"filename": safe_name, "error": str(e)})
            logger.error("batch_upload_error", filename=safe_name, error=str(e))

    # Invalidate cached source names so agent picks up new files immediately
    try:
        from backend.agents.graph import invalidate_source_cache
        invalidate_source_cache()
    except Exception:
        pass

    total = len(results) + len(errors)
    return MultiUploadResponse(
        status="success" if results else "failed",
        total_files=total,
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
        message=f"Processed {total} files: {len(results)} succeeded, {len(errors)} failed",
    )


# ── URL Ingest ─────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=UploadResponse)
async def ingest_from_url(request: IngestURLRequest):
    """
    Download a file from a URL and ingest it into the knowledge base.
    Supports: PDF, TXT, DOCX, JSON, CSV (structured, semi-structured, unstructured).
    """
    import httpx
    from urllib.parse import urlparse, unquote

    url = request.url.strip()

    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http:// and https:// URLs are supported.")

    # SSRF protection: block internal/private IPs
    _validate_url_safe(url)

    # Derive filename from URL path
    url_path = unquote(parsed.path).rstrip("/")
    raw_name = url_path.split("/")[-1] if url_path else "download"
    if not raw_name or raw_name == "":
        raw_name = "download"
    safe_name = _sanitise_filename(raw_name)

    # Ensure it has a supported extension
    from backend.ingestion.document_loader import supported_extensions
    ext = Path(safe_name).suffix.lower()
    if ext not in supported_extensions():
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(supported_extensions())}",
        )

    # Download
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=422, detail=f"URL returned HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=422, detail=f"Failed to download: {e}")

    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb} MB")

    # Save locally
    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # Ingest
    try:
        result = ingest_file(save_path)
    except ValueError as e:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        os.remove(save_path)
        logger.error("ingestion_error", filename=safe_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # Invalidate cached source names so agent picks up new file immediately
    try:
        from backend.agents.graph import invalidate_source_cache
        invalidate_source_cache()
    except Exception:
        pass

    return UploadResponse(
        status="success",
        filename=safe_name,
        source_id=result["source_id"],
        chunks_indexed=result["chunks_indexed"],
        file_type=result["file_type"],
        message=f"Successfully indexed {result['chunks_indexed']} chunks from {safe_name}",
    )


# ── Web Scrape ─────────────────────────────────────────────────────────────


@app.post("/scrape", response_model=UploadResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a web page (HTML) and ingest its text content.
    Extracts readable text from the page body, ignoring scripts/styles.
    """
    import hashlib
    import re
    import httpx
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup

    url = request.url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http:// and https:// URLs are supported.")

    # SSRF protection
    _validate_url_safe(url)

    # Fetch the page
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 AgenticRAG/1.0"})
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=422, detail=f"URL returned HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch: {e}")

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise HTTPException(status_code=422, detail=f"Expected HTML page, got content-type: {content_type}")

    # Parse HTML
    soup = BeautifulSoup(resp.text, "lxml")

    # Remove non-content elements (scripts, styles, nav, footer, sidebars, etc.)
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe",
                      "aside", "form", "button", "input", "select", "textarea"]):
        tag.decompose()

    # Remove common noise elements by exact CSS class / id matching.
    # We check individual class tokens (HTML classes are space-separated)
    # to avoid regex word-boundary issues (e.g. "toc" matching inside
    # "vector-feature-toc-pinned-clientpref-1" on the <html> tag).
    _noise_class_set = {
        "sidebar", "navbox", "navbox-styles", "printfooter",
        "mw-editsection", "mw-jump-link", "catlinks",
        "noprint",                      # Wikipedia UI chrome
    }
    _noise_id_set = {
        "sidebar", "toc", "catlinks", "mw-jump", "printfooter",
        "siteSub", "contentSub",
    }
    # Collect elements to remove first, then decompose (avoids mutating tree during iteration)
    to_remove = []
    for el in soup.find_all(attrs={"class": True}):
        try:
            cls_list = el.get("class", [])
            if _noise_class_set.intersection(cls_list):
                to_remove.append(el)
        except Exception:
            continue
    for el in soup.find_all(attrs={"id": True}):
        try:
            el_id = el.get("id", "")
            if el_id in _noise_id_set:
                to_remove.append(el)
        except Exception:
            continue
    for el in to_remove:
        try:
            el.decompose()
        except Exception:
            continue

    title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc

    # Extract text from the main content area (prefer <article> or <main> or #content)
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id="content")
        or soup.find(id="mw-content-text")  # Wikipedia
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    text = main_content.get_text(separator="\n", strip=True)

    # Clean up the extracted text
    # Remove Wikipedia-style reference markers like [1], [2], [edit], [citation needed]
    text = re.sub(r'\[(?:\d+|edit|citation needed|needs? update)\]', '', text)
    # Collapse multiple blank lines into double newlines (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lines that are just single characters or very short noise (e.g. "^", "v", "e")
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
    text = '\n'.join(cleaned_lines)

    if not text or len(text) < 50:
        raise HTTPException(status_code=422, detail="Page has too little text content to ingest.")

    # Save as .txt
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    safe_name = _sanitise_filename(f"{parsed.netloc}_{url_hash}.txt")

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {url}\nTitle: {title}\n\n{text}")

    try:
        result = ingest_file(save_path)
    except Exception as e:
        os.remove(save_path)
        logger.error("scrape_ingestion_error", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # Invalidate cached source names so agent picks up new file immediately
    try:
        from backend.agents.graph import invalidate_source_cache
        invalidate_source_cache()
    except Exception:
        pass

    return UploadResponse(
        status="success",
        filename=safe_name,
        source_id=result["source_id"],
        chunks_indexed=result["chunks_indexed"],
        file_type="scraped_html",
        message=f"Scraped '{title}' — {result['chunks_indexed']} chunks indexed",
    )


@app.get("/sources", response_model=list[SourceInfo])
async def list_sources():
    vs = get_vector_store()
    sources = _chroma_sources(vs)
    return [
        SourceInfo(
            source_id=s["filename"][:12],
            filename=s["filename"],
            file_type=s["file_type"],
            chunks=s["chunk_count"],
            uploaded_at="unknown",
        )
        for s in sources
    ]


@app.delete("/sources")
async def clear_all_sources():
    """Delete ALL data from the knowledge base (vector store, SQL tables, uploaded files)."""
    vs = get_vector_store()
    sql = get_sql_store()

    # Delete all vectors
    sources = _chroma_sources(vs)
    total_chunks = 0
    for s in sources:
        try:
            total_chunks += vs.delete_by_source(s["filename"])
        except Exception as e:
            logger.error("clear_source_error", filename=s["filename"], error=str(e))

    # Drop all SQL tables
    tables_dropped = 0
    try:
        tables_dropped = sql.drop_all_tables()
    except Exception as e:
        logger.error("clear_sql_error", error=str(e))

    # Remove uploaded files
    files_removed = 0
    upload_dir = settings.upload_dir
    if os.path.isdir(upload_dir):
        for f in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                files_removed += 1

    # Invalidate caches
    try:
        from backend.agents.graph import invalidate_source_cache
        invalidate_source_cache()
    except Exception:
        pass

    return {
        "status": "cleared",
        "chunks_deleted": total_chunks,
        "tables_dropped": tables_dropped,
        "files_removed": files_removed,
        "message": f"Knowledge base cleared: {total_chunks} chunks, {tables_dropped} tables, {files_removed} files removed.",
    }


@app.delete("/sources/{filename}")
async def delete_source(filename: str):
    """Delete all chunks belonging to a specific source file from the knowledge base."""
    vs = get_vector_store()

    # Check if source exists
    sources = _chroma_sources(vs)
    existing = [s for s in sources if s["filename"] == filename]
    if not existing:
        raise HTTPException(status_code=404, detail=f"Source '{filename}' not found in knowledge base.")

    try:
        deleted_count = vs.delete_by_source(filename)
    except Exception as e:
        logger.error("delete_source_error", filename=filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    # Also remove the uploaded file if it exists
    upload_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(upload_path):
        os.remove(upload_path)

    return {
        "status": "deleted",
        "filename": filename,
        "chunks_deleted": deleted_count,
        "message": f"Removed {deleted_count} chunks for '{filename}'",
    }


@app.get("/sessions/{session_id}", response_model=SessionHistory)
async def get_session(session_id: str):
    store = get_session_store()
    if not store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return store.get_history(session_id)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    store = get_session_store()
    store.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    store = get_session_store()
    return {"sessions": store.list_sessions()}


# ── Evaluation ─────────────────────────────────────────────────────────────


@app.get("/eval/health")
async def eval_health():
    """Check if the evaluation module is available."""
    try:
        from evaluation.metrics import evaluate_single
        return {"status": "available", "module": "evaluation.metrics"}
    except ImportError:
        return {"status": "unavailable", "detail": "Evaluation module not installed."}