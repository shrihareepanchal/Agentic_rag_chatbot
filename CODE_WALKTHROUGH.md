# Agentic RAG Chatbot — Complete Code Walkthrough

This document provides an end-to-end explanation of every file, function, and line of code in the Agentic RAG Chatbot project. It follows the data flow from configuration through document ingestion, retrieval, agent reasoning, and API response.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Configuration — `backend/config.py`](#2-configuration--backendconfigpy)
3. [Logging — `backend/utils/logger.py`](#3-logging--backendutilsloggerpy)
4. [Data Models — `backend/models/schemas.py`](#4-data-models--backendmodelsschemaspyy)
5. [Document Loading — `backend/ingestion/document_loader.py`](#5-document-loading--backendingestiondocument_loaderpy)
6. [Chunking — `backend/ingestion/chunker.py`](#6-chunking--backendingestionchunkerpy)
7. [Embedding — `backend/ingestion/embedder.py`](#7-embedding--backendingestionembedderpy)
8. [Indexing Pipeline — `backend/ingestion/indexer.py`](#8-indexing-pipeline--backendingestionindexerpy)
9. [Vector Store — `backend/retrieval/vector_store.py`](#9-vector-store--backendretrievalvector_storepy)
10. [SQL Store — `backend/retrieval/sql_store.py`](#10-sql-store--backendretrievalsql_storepy)
11. [Guardrails — Injection Detection — `backend/guardrails/injection_detector.py`](#11-guardrails--injection-detection--backendguardrailsinjection_detectorpy)
12. [Guardrails — PII Redaction — `backend/guardrails/pii_redactor.py`](#12-guardrails--pii-redaction--backendguardrailspii_redactorpy)
13. [Session Memory — `backend/memory/session_store.py`](#13-session-memory--backendmemorysession_storepy)
14. [Agent Tools — `backend/agents/tools.py`](#14-agent-tools--backendagentstoolspy)
15. [Agent Graph — `backend/agents/graph.py`](#15-agent-graph--backendagentsgraphpy)
16. [FastAPI Application — `backend/main.py`](#16-fastapi-application--backendmainpy)
17. [Frontend — `frontend/chat.html` & `frontend/index.html`](#17-frontend)
18. [Infrastructure — `Dockerfile`, `docker-compose.yml`, `Makefile`](#18-infrastructure)
19. [Data Flow Summary](#19-data-flow-summary)

---

## 1. Project Architecture Overview

```
User → FastAPI (main.py)
         ├── /upload, /ingest, /scrape  →  Indexer Pipeline
         │                                   ├── Document Loader
         │                                   ├── Chunker
         │                                   ├── Embedder
         │                                   └── Vector Store + SQL Store
         │
         └── /chat  →  LangGraph Agent (graph.py)
                         ├── input_guard_node  (PII + injection + rewrite)
                         ├── agent_node        (LLM reasoning with tools)
                         ├── tools_executor    (ToolNode runs tools)
                         ├── extract_citations (parse tool results)
                         └── output_guard_node (PII redaction + confidence)
```

The system is a **Retrieval-Augmented Generation (RAG)** chatbot that:
1. **Ingests** documents (PDF, TXT, DOCX, JSON, CSV, URLs, web scrapes)
2. **Chunks** them into semantically meaningful segments
3. **Embeds** them using OpenAI `text-embedding-3-small`
4. **Stores** them in ChromaDB (vector) and SQLite (structured)
5. **Retrieves** relevant chunks using semantic similarity + keyword matching
6. **Reasons** using a LangGraph state machine with GPT-4o-mini
7. **Guards** input/output with PII redaction and injection detection

---

## 2. Configuration — `backend/config.py`

**Purpose:** Centralized configuration management using `pydantic-settings`. All settings are loaded from environment variables (`.env` file).

```python
from __future__ import annotations
```
- Enables postponed evaluation of annotations (PEP 563), allowing forward references in type hints.

```python
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
```
- `os`: File system operations for directory creation.
- `lru_cache`: Memoizes the settings singleton so it's only created once.
- `Path`: Cross-platform path handling.
- `BaseSettings`: Pydantic class that auto-reads values from environment variables.

```python
BASE_DIR = Path(__file__).resolve().parent.parent
```
- Resolves to the project root (`/agentic-rag/`). `__file__` is `backend/config.py`, `.parent.parent` goes up two levels.

```python
class Settings(BaseSettings):
```
- Defines all application settings as typed class attributes. Pydantic validates types and provides defaults.

### Settings Fields (line-by-line):

```python
    openai_api_key: str = ""
```
- OpenAI API key. Read from `OPENAI_API_KEY` env var. Empty string means mock mode will activate.

```python
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
```
- Directory where ChromaDB stores its persistent SQLite database + embedding files. Defaults to `data/chroma_db/`.

```python
    upload_dir: str = str(BASE_DIR / "uploads")
```
- Directory where uploaded files are temporarily stored before processing.

```python
    log_level: str = "INFO"
```
- Logging level. Options: DEBUG, INFO, WARNING, ERROR. Controls verbosity of structlog output.

```python
    chunk_size: int = 512
    chunk_overlap: int = 64
```
- **chunk_size**: Maximum number of characters per text chunk during splitting. 512 characters provides a good balance between context and specificity.
- **chunk_overlap**: Number of overlapping characters between consecutive chunks. Prevents information loss at chunk boundaries.

```python
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
```
- **embedding_model**: OpenAI embedding model name. `text-embedding-3-small` is cost-efficient with good quality.
- **embedding_dimensions**: Output dimensionality of the embedding vectors. Must match the model's output size.

```python
    llm_model: str = "gpt-4o-mini"
```
- The language model used for agent reasoning. `gpt-4o-mini` balances speed, cost, and quality.

```python
    confidence_threshold: float = 0.25
```
- Minimum confidence score (0-1) for the system to trust its answer. Below this, it returns "I don't know".

```python
    retrieval_top_k: int = 10
```
- Number of most-similar chunks to retrieve from ChromaDB during semantic search.

```python
    mock_mode: bool = False
```
- When `True`, uses mock LLM and mock embeddings for testing without an API key.

```python
    enable_pii_redaction: bool = True
    enable_injection_guard: bool = True
```
- Feature flags for the input/output guardrails. Both default to enabled.

```python
    max_file_size_mb: int = 50
```
- Maximum upload file size in megabytes. Files larger than this are rejected to prevent memory issues.

```python
    agent_max_iterations: int = 10
```
- Maximum number of reasoning loops the agent can take (LLM→tools→LLM cycles). Prevents infinite loops.

```python
    sql_db_path: str = str(BASE_DIR / "data" / "structured.db")
```
- Path to the SQLite database file where CSV/JSON structured data is stored for SQL queries.

```python
    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"
```
- **env_file**: Tells Pydantic to load from the `.env` file at project root.
- **env_file_encoding**: UTF-8 encoding for the env file.
- **extra = "ignore"**: Silently ignores unknown environment variables instead of raising errors.

```python
def ensure_dirs(settings: Settings) -> None:
    for d in (settings.chroma_persist_dir, settings.upload_dir, str(BASE_DIR / "logs")):
        os.makedirs(d, exist_ok=True)
```
- Creates the `chroma_db/`, `uploads/`, and `logs/` directories if they don't exist. Called once during startup.

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    ensure_dirs(s)
    return s
```
- **Singleton pattern**: `lru_cache(maxsize=1)` ensures only one `Settings` instance exists across the entire application. Every module calls `get_settings()` to access configuration.

---

## 3. Logging — `backend/utils/logger.py`

**Purpose:** Structured JSON logging with request-tracing support using `structlog`.

```python
from __future__ import annotations
import logging
import structlog
from contextvars import ContextVar
```
- `structlog`: Library for structured, key-value log output (JSON format).
- `ContextVar`: Python's async-safe thread-local storage. Allows attaching a request ID to all log entries within a single request.

```python
_request_id: ContextVar[str] = ContextVar("request_id", default="no-request")
```
- Holds the current request's UUID. Each incoming HTTP request sets this via middleware, and it's automatically attached to every log line.

```python
def set_request_id(rid: str) -> None:
    _request_id.set(rid)
```
- Called by FastAPI middleware to set the request ID for the current async context.

```python
def add_request_id(logger, method_name, event_dict):
    event_dict["request_id"] = _request_id.get()
    return event_dict
```
- **structlog processor**: Injects the `request_id` into every log event dictionary. This means every log entry from any module automatically includes which request triggered it.

```python
def add_severity(logger, method_name, event_dict):
    event_dict["severity"] = method_name.upper()
    return event_dict
```
- Converts the log method name (e.g., `info`, `warning`) to uppercase `severity` field in the output JSON.

```python
def configure_logging(level: str = "INFO") -> None:
```
- Sets up the entire logging pipeline. Called once from `main.py` during app startup.

```python
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```
- Configures Python's standard `logging`. Suppresses noisy Uvicorn access logs to WARNING level.

```python
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            add_request_id,
            add_severity,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        ...
    )
```
- **Processor pipeline** (each function transforms the log event in sequence):
  1. `merge_contextvars`: Merges any context variables into the log event.
  2. `add_log_level`: Adds the log level as a field.
  3. `add_request_id`: Adds the current request's UUID.
  4. `add_severity`: Adds uppercase severity.
  5. `TimeStamper`: Adds ISO 8601 timestamp.
  6. `StackInfoRenderer`: Includes stack traces when available.
  7. `format_exc_info`: Formats exception info.
  8. `JSONRenderer`: Final output as a JSON string.

```python
def get_logger(name: str = "agentic_rag") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```
- Factory function for getting a named logger. All modules call `get_logger(__name__)` or `get_logger()`.

---

## 4. Data Models — `backend/models/schemas.py`

**Purpose:** Pydantic models that define the shape of all API requests and responses. These are used by FastAPI for automatic validation, serialization, and OpenAPI documentation.

### Core Request/Response Models:

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = False
    active_sources: list[str] | None = None
```
- **message**: The user's question text (required).
- **session_id**: Optional UUID; if omitted, a new session is created.
- **stream**: Reserved for streaming responses (not yet implemented).
- **active_sources**: List of filenames the user has uploaded in this session. The agent uses this to prioritize retrieval from these files.

```python
class Citation(BaseModel):
    source: str
    chunk_id: str
    page_number: int | None = None
    score: float = 0.0
    snippet: str = ""
```
- Represents a retrieved document chunk cited in the answer.
- **source**: Original filename (e.g., `report.pdf`).
- **chunk_id**: Unique ID like `report.pdf::chunk_3`.
- **page_number**: Page number for PDFs (None for other formats).
- **score**: Relevance score from 0 to 1 (after RRF re-ranking).
- **snippet**: Short excerpt of the chunk text.

```python
class ToolCall(BaseModel):
    tool: str
    input: str
    output_summary: str
```
- Records which tools the agent invoked during reasoning.

```python
class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    follow_up_suggestions: list[str] = []
    confidence: float = 1.0
    pii_detected: bool = False
    injection_detected: bool = False
    low_confidence: bool = False
```
- The full response sent back to the user.
- **follow_up_suggestions**: Up to 3 AI-generated contextual follow-up questions.
- **confidence**: Weighted score (0-1) based on citation relevance.
- **low_confidence**: Flag when confidence is below the threshold AND citations exist.

```python
class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str
    source_id: str = ""
```
- Response after file upload/ingestion. Reports how many chunks were created.

```python
class SourceInfo(BaseModel):
    name: str
    chunk_count: int
    source_type: str = "unknown"
```
- Information about an indexed source document.

```python
class IngestURLRequest(BaseModel):
    url: str
    source_name: str = ""
```
- Request to ingest a document from a URL.

```python
class ScrapeRequest(BaseModel):
    url: str
    source_name: str = ""
```
- Request to scrape and ingest a web page.

```python
class ConversationMessage(BaseModel):
    role: str
    content: str
```
- Single message in session history (role is "human" or "ai").

```python
class SessionHistory(BaseModel):
    session_id: str
    messages: list[ConversationMessage]
    message_count: int
```
- Full conversation history for a session.

---

## 5. Document Loading — `backend/ingestion/document_loader.py`

**Purpose:** Load various file formats into `(text, metadata)` tuples ready for chunking.

### Constants:

```python
MIN_CHUNK_LENGTH = 20
```
- Minimum character length for a text fragment to be kept. Shorter fragments are merged with adjacent ones to avoid tiny useless chunks.

### `load_pdf(path: str) -> List[Tuple[str, Dict]]`

```python
def load_pdf(path: str) -> List[Tuple[str, Dict]]:
    reader = PdfReader(path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append((text, {
                "source": Path(path).name,
                "page_number": i + 1,
                "doc_type": "pdf",
            }))
    return documents
```
- Uses `pypdf.PdfReader` to extract text page-by-page.
- Each page becomes a separate document with its page number in metadata.
- Empty pages are skipped.

### `load_txt(path: str) -> List[Tuple[str, Dict]]`

```python
def load_txt(path: str) -> List[Tuple[str, Dict]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return []
    ext = Path(path).suffix.lower()
    return [(text, {
        "source": Path(path).name,
        "doc_type": "markdown" if ext == ".md" else "text",
    })]
```
- Reads the entire file as UTF-8 text.
- Handles `.md` files by setting `doc_type` to "markdown".
- Returns the whole file as a single document (chunking happens later).

### `load_docx(path: str) -> List[Tuple[str, Dict]]`

```python
def load_docx(path: str) -> List[Tuple[str, Dict]]:
    doc = DocxDocument(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
```
- Opens `.docx` files using `python-docx`.

```python
    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                paragraphs.append(" | ".join(row_texts))
```
- Extracts table contents by joining cell values with ` | ` separators.

```python
    # Skip very short paragraphs by joining them with neighbors
    merged = []
    buffer = ""
    for p in paragraphs:
        if len(p) < MIN_CHUNK_LENGTH:
            buffer += " " + p
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(p)
    if buffer:
        merged.append(buffer.strip())
```
- **Smart merging**: Short paragraphs (< 20 chars) are concatenated into a buffer instead of becoming standalone paragraphs. This prevents tiny, meaningless chunks.

```python
    full_text = "\n\n".join(merged)
    return [(full_text, {"source": Path(path).name, "doc_type": "docx"})]
```
- Joins all merged paragraphs with double newlines and returns as a single document.

### `load_json(path: str) -> List[Tuple[str, Dict]]`

```python
def load_json(path: str) -> List[Tuple[str, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
```

**For list of objects** (e.g., `[{"name": "Alice"}, {"name": "Bob"}]`):
```python
    if isinstance(data, list):
        documents = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                text = "\n".join(f"{k}: {v}" for k, v in item.items())
            else:
                text = str(item)
            documents.append((text, {
                "source": Path(path).name,
                "doc_type": "json",
                "item_index": idx,
            }))
        return documents
```
- Each item in the list becomes a separate document. Dictionary items are expanded to `key: value` lines.

**For dictionaries** (e.g., `{"section1": {...}, "section2": {...}}`):
```python
    if isinstance(data, dict):
        documents = []
        items = list(data.items())
        group_size = 5
        for i in range(0, len(items), group_size):
            group = items[i:i + group_size]
            text = "\n".join(f"{k}: {json.dumps(v, default=str)}" for k, v in group)
            documents.append((text, {
                "source": Path(path).name,
                "doc_type": "json",
                "group_start": i,
            }))
        return documents
```
- Keys are grouped in batches of 5 for manageable chunk sizes. Each group is serialized to text.

### `load_csv(path: str) -> List[Tuple[str, Dict]]`

```python
def load_csv(path: str) -> List[Tuple[str, Dict]]:
    documents = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if text.strip():
                documents.append((text, {
                    "source": Path(path).name,
                    "doc_type": "csv",
                    "row_index": idx,
                }))
    return documents
```
- Each CSV row becomes a separate document. Column headers become keys in `key: value` format.
- Empty values are filtered out.

### `load_file(path: str) -> List[Tuple[str, Dict]]`

```python
def load_file(path: str) -> List[Tuple[str, Dict]]:
    ext = Path(path).suffix.lower()
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
    return loader(path)
```
- **Dispatcher function**: Maps file extensions to loader functions. `.md` reuses the `.txt` loader.

### `supported_extensions()`

```python
def supported_extensions() -> List[str]:
    return [".pdf", ".txt", ".md", ".docx", ".json", ".csv"]
```
- Returns the list of supported file extensions, used by the upload endpoint for validation.

---

## 6. Chunking — `backend/ingestion/chunker.py`

**Purpose:** Split loaded documents into smaller text chunks suitable for embedding and retrieval. Uses LangChain's `RecursiveCharacterTextSplitter` with contextual metadata prefixes.

### `_build_context_prefix(metadata: Dict) -> str`

```python
def _build_context_prefix(metadata: Dict) -> str:
    parts = []
    if metadata.get("source"):
        parts.append(f"Document: {metadata['source']}")
    if metadata.get("doc_type"):
        parts.append(f"Type: {metadata['doc_type']}")
    if metadata.get("page_number"):
        parts.append(f"Page: {metadata['page_number']}")
    return f"[{' | '.join(parts)}] " if parts else ""
```
- Builds a human-readable prefix like `[Document: report.pdf | Type: pdf | Page: 3]`.
- This prefix is prepended to each chunk text, providing context to the embedding model and the LLM about where the chunk came from.

### `chunk_documents(documents, chunk_size, chunk_overlap) -> List[Dict]`

```python
def chunk_documents(
    documents: List[Tuple[str, Dict]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
```

```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
```
- **RecursiveCharacterTextSplitter**: Tries to split on the largest separator first (`\n\n` = paragraph break), falling back to smaller ones. This preserves semantic boundaries.
- **Separator hierarchy**: Paragraph → Line → Sentence → Word → Character.

```python
    chunks = []
    global_idx = 0
    for text, metadata in documents:
        context_prefix = _build_context_prefix(metadata)
        splits = splitter.split_text(text)
```
- For each document, builds the context prefix and splits the text.

```python
        for split_text in splits:
            chunk_text = f"{context_prefix}{split_text}"
            chunk_id = f"{metadata.get('source', 'unknown')}::chunk_{global_idx}"
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": global_idx},
            }
            chunks.append(chunk)
            global_idx += 1
```
- Each split gets:
  - **Context prefix prepended** to the text.
  - A **unique chunk_id** like `report.pdf::chunk_42`.
  - All original metadata plus a `chunk_index`.
  - The `global_idx` counter is shared across all documents, ensuring unique IDs.

```python
    logger.info("chunking_complete", total_chunks=len(chunks), ...)
    return chunks
```

---

## 7. Embedding — `backend/ingestion/embedder.py`

**Purpose:** Convert text chunks into dense numerical vectors for similarity search. Provides a real OpenAI embedder and a mock fallback.

### `MockEmbedder`

```python
class MockEmbedder:
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
```

```python
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            hash_bytes = hashlib.md5(text.encode()).digest()
            rng = random.Random(int.from_bytes(hash_bytes[:4], "big"))
            raw = [rng.gauss(0, 1) for _ in range(self.dimensions)]
            norm = math.sqrt(sum(x * x for x in raw))
            normalized = [x / norm for x in raw]
            embeddings.append(normalized)
        return embeddings
```
- **Deterministic mock**: Uses MD5 hash of the text as a seed for the random number generator. Same text always produces the same embedding vector.
- **Unit-norm normalization**: Divides each component by the L2 norm, producing unit vectors suitable for cosine similarity.
- Used when `mock_mode=True` or when the OpenAI API key is missing.

### `RealEmbedder`

```python
class RealEmbedder:
    _BATCH_SIZE = 100

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAIEmbeddings(model=model, dimensions=dimensions)
```
- Wraps LangChain's `OpenAIEmbeddings` which calls the OpenAI API.
- **_BATCH_SIZE = 100**: Processes embeddings in batches of 100 texts to avoid API rate limits and payload size limits.

```python
    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i:i + self._BATCH_SIZE]
            batch_embeddings = self._client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
```
- Iterates through texts in batches and accumulates all embedding vectors.

### `get_embedder()` Factory

```python
def get_embedder():
    settings = get_settings()
    if settings.mock_mode:
        return MockEmbedder(dimensions=settings.embedding_dimensions)
    try:
        return RealEmbedder(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    except Exception as e:
        logger.warning("real_embedder_failed_falling_back_to_mock", error=str(e))
        return MockEmbedder(dimensions=settings.embedding_dimensions)
```
- Tries to create a `RealEmbedder` first. Falls back to `MockEmbedder` on any error (e.g., missing API key).

---

## 8. Indexing Pipeline — `backend/ingestion/indexer.py`

**Purpose:** Orchestrates the full ingestion pipeline: dedup → load → chunk → embed → store.

### `_make_source_id(filename: str) -> str`

```python
def _make_source_id(filename: str) -> str:
    return hashlib.sha1(filename.encode()).hexdigest()[:12]
```
- Creates a short (12-char) deterministic ID from the filename using SHA-1. Used as a reference identifier.

### `_load_into_sql(file_path: str, file_name: str)`

```python
def _load_into_sql(file_path: str, file_name: str) -> None:
    ext = Path(file_path).suffix.lower()
    if ext not in (".csv", ".json"):
        return
```
- Only processes CSV and JSON files for SQL storage.

```python
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", Path(file_name).stem).lower()
```
- Creates a SQL-safe table name from the filename. Special characters become underscores.

```python
    sql_store = get_sql_store()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return
    df.to_sql(table_name, sql_store.engine, if_exists="replace", index=False)
```
- Loads data into pandas DataFrame and writes to SQLite using `to_sql()`.
- `if_exists="replace"`: Overwrites existing table if re-ingesting the same file.

### `ingest_file(file_path: str, original_filename: str | None = None) -> int`

This is the main pipeline function:

```python
def ingest_file(file_path: str, original_filename: str | None = None) -> int:
    file_name = original_filename or Path(file_path).name
    source_id = _make_source_id(file_name)
```

**Step 1 — Deduplication:**
```python
    store = get_vector_store()
    store.delete_by_source(file_name)
```
- Deletes all existing chunks from ChromaDB for this filename. This prevents duplicate entries when re-uploading the same file.

**Step 2 — Load:**
```python
    documents = load_file(file_path)
```
- Calls the appropriate loader based on file extension.

**Step 3 — Chunk:**
```python
    settings = get_settings()
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
```
- Splits documents into overlapping chunks of 512 characters.

**Step 4 — Embed:**
```python
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)
```
- Generates embedding vectors for all chunk texts.

**Step 5 — Store:**
```python
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
```
- Stores chunk IDs, texts, embeddings, and metadata in ChromaDB.

**SQL loading:**
```python
    _load_into_sql(file_path, file_name)
```
- Additionally loads CSV/JSON into SQLite for structured queries.

```python
    return len(chunks)
```
- Returns the number of chunks created.

---

## 9. Vector Store — `backend/retrieval/vector_store.py`

**Purpose:** Adapter around ChromaDB providing add, search, delete, and collection management operations.

### `ChromaVectorStoreAdapter`

```python
class ChromaVectorStoreAdapter:
    def __init__(self, persist_dir: str, collection_name: str = "documents",
                 embedding_model: str = "text-embedding-3-small"):
```
- Wraps LangChain's `Chroma` vector store with a consistent API.

```python
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=self._embeddings,
        )
```
- Creates a persistent Chroma collection backed by files on disk.

### Key Methods:

**`add(ids, texts, embeddings, metadatas)`:**
```python
    def add(self, ids, texts, embeddings=None, metadatas=None):
        collection = self._store._collection
        if embeddings is not None:
            # Direct add with pre-computed embeddings — skips OpenAI call
            BATCH = 5000
            for i in range(0, len(ids), BATCH):
                collection.add(
                    ids=ids[i:i+BATCH],
                    documents=texts[i:i+BATCH],
                    embeddings=embeddings[i:i+BATCH],
                    metadatas=metadatas[i:i+BATCH] if metadatas else None,
                )
        else:
            self._store.add_texts(texts=texts, ids=ids, metadatas=metadatas)
```
- When embeddings are pre-computed (the normal path via the indexer), bypasses LangChain and adds directly to ChromaDB's collection in batches of 5000. This avoids redundant embedding API calls.

**`search(query, k, source_filter)`:**
```python
    def search(self, query: str, k: int = 5, source_filter: str | None = None):
        kwargs = {}
        if source_filter:
            kwargs["filter"] = {"source": source_filter}
        results = self._store.similarity_search_with_relevance_scores(query, k=k, **kwargs)
```
- Performs semantic search using the query text. ChromaDB embeds the query using the same OpenAI model and finds the nearest neighbors.
- Optional **source_filter** restricts search to a specific filename.

**`delete_by_source(source_name)`:**
```python
    def delete_by_source(self, source_name: str):
        collection = self._store._collection
        result = collection.get(where={"source": source_name})
        ids_to_delete = result.get("ids", [])
        BATCH = 5000
        for i in range(0, len(ids_to_delete), BATCH):
            collection.delete(ids=ids_to_delete[i:i+BATCH])
```
- Finds all chunk IDs with matching source metadata and deletes them in batches.

### Singleton:

```python
@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStoreAdapter:
    settings = get_settings()
    return ChromaVectorStoreAdapter(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )
```
- Single shared instance across the application.

---

## 10. SQL Store — `backend/retrieval/sql_store.py`

**Purpose:** SQLite-based store for structured CSV/JSON data, enabling SQL queries via the agent.

### Security:

```python
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
```
- Regex pattern that blocks destructive SQL keywords. Only `SELECT` queries are allowed.

### `SQLStore`

```python
class SQLStore:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
```
- Creates a SQLAlchemy engine connected to the SQLite database file.

**`execute_query(sql: str) -> List[Dict]`:**
```python
    def execute_query(self, sql: str) -> List[Dict]:
        sql_stripped = sql.strip().rstrip(";")
        if _FORBIDDEN_SQL.search(sql_stripped):
            raise ValueError("Only SELECT queries are allowed.")
        if not sql_stripped.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
```
- Double validation: regex check AND prefix check ensure only SELECT statements execute.

```python
        with self.engine.connect() as conn:
            result = conn.execute(text(sql_stripped))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return rows
```
- Executes the query and returns results as a list of dictionaries.

**`get_schema() -> Dict`:**
```python
    def get_schema(self) -> Dict:
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [{"name": c["name"], "type": str(c["type"])} for c in columns]
        return schema
```
- Uses SQLAlchemy's inspector to enumerate all tables and their columns with types.

---

## 11. Guardrails — Injection Detection — `backend/guardrails/injection_detector.py`

**Purpose:** Detect prompt injection attempts in user input using regex pattern matching with risk scoring.

### Injection Rules:

```python
_INJECTION_RULES: List[_InjectionRule] = [
    _InjectionRule("ignore_previous", re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.I), 0.9),
    _InjectionRule("role_switch", re.compile(r"you\s+are\s+now\s+(a|an|the)?\s*\w+", re.I), 0.8),
    _InjectionRule("jailbreak_dan", re.compile(r"\bDAN\b|do anything now|jailbreak", re.I), 0.95),
    _InjectionRule("system_prompt_leak", re.compile(r"(show|reveal|print|output)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.I), 0.85),
    _InjectionRule("delimiter_injection", re.compile(r"```\s*(system|assistant|instruction)", re.I), 0.6),
    _InjectionRule("base64_payload", re.compile(r"[A-Za-z0-9+/]{40,}={0,2}", re.I), 0.4),
    _InjectionRule("command_execution", re.compile(r"(run|execute|eval|exec)\s*\(", re.I), 0.9),
]
```
- Each rule has a **name**, **regex pattern**, and **risk score** (0 to 1).
- Higher scores indicate more dangerous injection attempts.

### Detection Logic:

```python
_INJECTION_THRESHOLD = 0.5
```

```python
def detect_injection(text: str) -> InjectionResult:
    triggered = []
    max_risk = 0.0
    for rule in _INJECTION_RULES:
        if rule.pattern.search(text):
            triggered.append(rule.name)
            max_risk = max(max_risk, rule.risk_score)
```
- Scans input text against all 7 rules. Tracks the highest risk score.

```python
    is_injection = max_risk >= _INJECTION_THRESHOLD
```
- If the maximum risk score exceeds 0.5, the input is flagged as an injection attempt.

```python
    sanitised = text
    if is_injection:
        for rule in _INJECTION_RULES:
            sanitised = rule.pattern.sub("[BLOCKED]", sanitised)
```
- Replaces matched patterns with `[BLOCKED]` in the sanitized text.

```python
    return InjectionResult(
        is_injection=is_injection,
        risk_score=max_risk,
        triggered_rules=triggered,
        sanitised_text=sanitised,
    )
```

---

## 12. Guardrails — PII Redaction — `backend/guardrails/pii_redactor.py`

**Purpose:** Detect and redact Personally Identifiable Information from text using regex patterns and optionally Microsoft Presidio.

### Regex Patterns:

```python
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE_US": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DATE_OF_BIRTH": re.compile(r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"),
}
```
- Six pattern types covering the most common PII categories.

### `PIIRedactor`

```python
class PIIRedactor:
    def __init__(self, use_presidio: bool = False):
        self._use_presidio = use_presidio
        self._presidio_analyzer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                self._presidio_analyzer = AnalyzerEngine()
            except ImportError:
                logger.warning("presidio_not_installed")
```
- Optionally loads Microsoft Presidio for deeper PII detection (NLP-based). Falls back gracefully if not installed.

**`detect_and_redact(text: str) -> Tuple[str, List[PIIMatch], bool]`:**
```python
    def detect_and_redact(self, text):
        matches = []
        # Regex detection
        for entity_type, pattern in _PII_PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                ))
```
- Finds all regex matches across all PII patterns.

```python
        # Redact in reverse order to preserve positions
        redacted = text
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            redacted = redacted[:match.start] + f"[{match.entity_type}]" + redacted[match.end:]
```
- Replaces PII with type labels like `[EMAIL]`, `[SSN]`. Processes in reverse order so character positions remain valid.

```python
        return redacted, matches, bool(matches)
```

---

## 13. Session Memory — `backend/memory/session_store.py`

**Purpose:** Thread-safe in-memory storage for conversation history across sessions.

### Constants:

```python
_MAX_HISTORY = 20
```
- Maximum number of messages (human + AI) to keep per session. Older messages are dropped from the front (FIFO).

### `SessionStore`

```python
class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._lock = Lock()
```
- Dictionary mapping `session_id → list of messages`.
- `threading.Lock` ensures thread safety for concurrent requests.

**`add_message(session_id, message)`:**
```python
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append(message)
            if len(self._sessions[session_id]) > _MAX_HISTORY:
                self._sessions[session_id] = self._sessions[session_id][-_MAX_HISTORY:]
```
- Appends the message and trims to the last 20 messages if the limit is exceeded.

**`get_history(session_id) -> List[BaseMessage]`:**
```python
    def get_history(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))
```
- Returns a copy of the message list (prevents external mutation).

### Singleton:

```python
_store: SessionStore | None = None

def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
```

---

## 14. Agent Tools — `backend/agents/tools.py`

**Purpose:** Defines 5 LangChain tools that the agent can invoke during reasoning. These tools interface with the vector store, SQL store, and provide utility functions.

### Reciprocal Rank Fusion (RRF)

```python
def _reciprocal_rank_fusion(
    semantic_results: List[Dict],
    query: str,
    k_constant: int = 60,
) -> List[Dict]:
```
- **RRF** is a technique from information retrieval that combines multiple ranking signals.
- It takes the semantic search results and re-ranks them by combining:
  1. **Semantic rank** (position in the vector search results)
  2. **Keyword rank** (based on word overlap between query and chunk text)

```python
    query_terms = set(re.findall(r'\w+', query.lower()))
```
- Extracts individual words from the query for keyword matching.

```python
    for item in semantic_results:
        chunk_text = item.get("snippet", "").lower()
        chunk_terms = set(re.findall(r'\w+', chunk_text))
        overlap = len(query_terms & chunk_terms)
        keyword_rank_map[item_id] = overlap_order
```
- Counts how many query words appear in each chunk's text.

```python
    for item_id in all_ids:
        s_rank = semantic_rank_map.get(item_id, len(semantic_results))
        k_rank = keyword_rank_map.get(item_id, len(semantic_results))
        rrf_score = (1.0 / (k_constant + s_rank)) + (1.0 / (k_constant + k_rank))
```
- **RRF formula**: `score = 1/(k+rank_semantic) + 1/(k+rank_keyword)`. The `k_constant=60` prevents any single ranking from dominating.

### Tool 1: `retrieve_documents`

```python
@tool
def retrieve_documents(query: str, source_filter: str = "") -> str:
    """Retrieve relevant document chunks from the knowledge base using semantic search."""
```
- The primary retrieval tool. The agent calls this when it needs to find information from uploaded documents.

Key logic:
```python
    results = store.search(query=query, k=settings.retrieval_top_k, source_filter=source_filter or None)
```
- Searches ChromaDB for the top-k most similar chunks.

```python
    formatted = _reciprocal_rank_fusion(formatted, query)
```
- Re-ranks results using RRF.

**Query mismatch detection:**
```python
    query_words = set(re.findall(r'\w{3,}', query.lower())) - _STOP_WORDS
    for r in formatted:
        chunk_words = set(re.findall(r'\w{3,}', r.get("snippet", "").lower()))
        overlap = query_words & chunk_words
        match_ratio = len(overlap) / max(len(query_words), 1)
        if match_ratio < 0.60 and r["score"] < 0.35:
            r["mismatch_warning"] = "Low keyword overlap with query"
```
- If less than 60% of query words appear in a chunk AND the semantic score is below 0.35, adds a warning. This helps the agent recognize when results may not be relevant.

### Tool 2: `get_document_chunks`

```python
@tool
def get_document_chunks(source_name: str) -> str:
    """Retrieve ALL text chunks from a specific document for summarization."""
```
- Used when the agent needs to summarize an entire document.

```python
    MAX_SNIPPET_CHARS = 1200
    max_chunks = 150
```
- Limits snippet size and chunk count for large documents.

```python
    collection = store._store._collection
    result = collection.get(where={"source": source_name}, limit=cap, include=["documents", "metadatas"])
```
- Directly queries ChromaDB collection for all chunks from the specified source.

### Tool 3: `get_database_schema`

```python
@tool
def get_database_schema(table_name: str = "") -> str:
    """Get the schema of SQL tables loaded from CSV/JSON files."""
```
- Returns table names and their column definitions. Helps the agent understand the structure before writing SQL queries.

### Tool 4: `query_database`

```python
@tool
def query_database(sql_query: str) -> str:
    """Execute a read-only SQL query against the structured data store."""
```

```python
    rows = sql_store.execute_query(sql_query)
    if len(rows) > 50:
        rows = rows[:50]
        truncated = True
```
- Executes SELECT queries with a 50-row limit. Catches and returns errors for the agent to adjust its query.

### Tool 5: `request_clarification`

```python
@tool
def request_clarification(question: str) -> str:
    """Ask the user for clarification when the query is ambiguous."""
```
- Returns a JSON response with `clarification_needed: true` that the extract_citations node detects, setting the state flag.

### `ALL_TOOLS`

```python
ALL_TOOLS = [
    retrieve_documents,
    get_document_chunks,
    get_database_schema,
    query_database,
    request_clarification,
]
```
- The list of tools bound to the agent LLM.

---

## 15. Agent Graph — `backend/agents/graph.py`

**Purpose:** The core reasoning engine. Implements a 5-node LangGraph state machine that processes user queries through guard rails, LLM reasoning with tool use, citation extraction, and output post-processing.

### Token Management Constants:

```python
_MAX_HISTORY_TOKENS = 60000
_SYSTEM_PROMPT_BUFFER = 4000
_RESPONSE_BUFFER = 4000
```
- **_MAX_HISTORY_TOKENS**: Maximum tokens for conversation history in the context window. Set high to preserve long conversations.
- **_SYSTEM_PROMPT_BUFFER**: Reserved tokens for the system prompt.
- **_RESPONSE_BUFFER**: Reserved tokens for the LLM's response.

### Source Name Caching:

```python
_source_cache_names: List[str] | None = None
_source_cache_time: float = 0.0
_SOURCE_CACHE_TTL = 30.0
```

```python
def _get_cached_source_names() -> List[str]:
    global _source_cache_names, _source_cache_time
    now = time.time()
    if _source_cache_names is not None and (now - _source_cache_time) < _SOURCE_CACHE_TTL:
        return _source_cache_names
    # ... fetch from ChromaDB and cache
```
- Caches the list of indexed document names for 30 seconds. Avoids querying ChromaDB on every chat request.

```python
def invalidate_source_cache():
    global _source_cache_names
    _source_cache_names = None
```
- Called after document upload/deletion to force a fresh fetch.

### Token Counting:

```python
_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    return _tokenizer
```
- Lazy-loads the tokenizer once. Uses tiktoken (OpenAI's tokenizer) for accurate token counting.

```python
def _count_tokens(text: str) -> int:
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4  # Fallback: ~4 chars per token
```
- Counts tokens using tiktoken. Falls back to character-based estimate on error.

### Message Trimming:

```python
def _trim_messages_to_token_limit(messages: List[BaseMessage]) -> List[BaseMessage]:
```
- Trims conversation history from the front (oldest first) to fit within `_MAX_HISTORY_TOKENS`.
- **Preserves atomic tool blocks**: Tool call messages and their corresponding ToolMessage responses are kept together. If one would be cut, both are removed.

### `AgentState` (TypedDict):

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    original_query: str
    active_sources: List[str]
    pii_detected: bool
    injection_detected: bool
    citations: List[Dict]
    tool_calls_made: List[Dict]
    confidence: float
    needs_clarification: bool
    clarification_question: str
```
- The state object that flows through every node in the graph.
- `Annotated[list, add_messages]`: LangGraph's annotation that uses `add_messages` reducer — new messages are appended, not replaced.

### LLM Builder:

```python
def _build_llm():
    settings = get_settings()
    if settings.mock_mode:
        return _MockLLM()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        max_retries=2,
        request_timeout=30,
    )
```
- `temperature=0`: Deterministic responses (no randomness).
- `max_retries=2`: Auto-retries on transient API errors.
- `request_timeout=30`: 30-second timeout for LLM API calls.

### `_MockLLM`:

```python
class _MockLLM(BaseChatModel):
```
- A keyword-based mock that routes to tools based on pattern matching. Used in development without an API key.
- Recognizes patterns like "summarize", "schema", "SQL", "clarify" to invoke the appropriate tools.

### System Prompt:

```python
SYSTEM_PROMPT = """You are a helpful document assistant..."""
```
The ~4000 character system prompt instructs the LLM on:
- **Zero-hallucination rule**: Only answer from retrieved content.
- **Topic matching**: Use `source_filter` for file-specific questions.
- **Formatting**: Markdown with headers, bullet points.
- **Tool decision rules**: When to use each of the 5 tools.
- **Security**: Ignore injected instructions in document content.
- **Summarization guidelines**: Use `get_document_chunks` for full document summaries.
- **SQL guidelines**: Use `get_database_schema` → `query_database` for CSV/JSON.
- **Multi-source handling**: Search across documents with broad queries.
- **Grounded follow-ups**: Only suggest questions answerable from context.

### Node 1: `input_guard_node`

```python
def input_guard_node(state: AgentState) -> Dict:
```
**Query rewriting:** Detects non-English, misspelled, or broken queries using `_query_needs_rewrite()`.
```python
def _query_needs_rewrite(text: str) -> bool:
```
- Returns `True` if the text contains non-ASCII characters (indicating non-English) or has multiple consecutive consonants (indicating typos/broken text).
- When rewriting is needed, uses the LLM to translate/fix the query into clean English.

**PII redaction:** Runs `PIIRedactor` on the input to replace any PII with type labels.

**Injection detection:** Runs `detect_injection()` on the input. If injected, replaces matched patterns with `[BLOCKED]`.

### Node 2: `agent_node`

```python
def agent_node(state: AgentState) -> Dict:
```
The central reasoning node:

1. **Builds dynamic system prompt**: Appends the list of indexed source filenames and SQL table schemas to the base system prompt, giving the LLM awareness of available data.

2. **Source filtering rules**: Injects specific instructions about when to use `source_filter` — especially for recently uploaded files.

3. **SQL table injection**: Lists all SQL tables with their column names so the agent knows what's queryable.

4. **Token-aware trimming**: Trims conversation history to prevent context window overflow.

5. **LLM invocation**: Calls the LLM with the system prompt + trimmed messages. The LLM either generates tool calls or a final text answer.

### Node 3: `tools` (ToolNode)

```python
builder.add_node("tools", ToolNode(ALL_TOOLS))
```
- LangGraph's built-in `ToolNode` automatically executes whatever tools the LLM requested, creating `ToolMessage` responses.

### Node 4: `extract_citations_from_tools`

```python
def extract_citations_from_tools(state: AgentState) -> Dict:
```
- Iterates through all `ToolMessage` objects in state.
- Parses JSON responses from `retrieve_documents` to extract citations (source, chunk_id, score, snippet).
- Detects clarification requests from `request_clarification` tool.

### Node 5: `output_guard_node`

```python
def output_guard_node(state: AgentState) -> Dict:
```
1. **Output PII redaction**: Runs `PIIRedactor` on the LLM's final answer to catch any PII that leaked from documents.

2. **Citation filtering**: Removes citations with scores below 0.10 (noise).

3. **Confidence calculation**:
   ```python
   confidence = round(0.6 * max_score + 0.4 * avg_score, 3)
   ```
   - Weighted formula: 60% best match + 40% average. The best match matters most because one highly relevant chunk can provide a good answer.

4. **Low confidence override**: If confidence is extremely low AND no meaningful content was found, replaces the answer with a "I don't know" message including the actual scores.

### Graph Compilation:

```python
def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("extract_citations", extract_citations_from_tools)
    builder.add_node("output_guard", output_guard_node)

    builder.add_edge(START, "input_guard")
    builder.add_edge("input_guard", "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "output_guard"})
    builder.add_edge("tools", "extract_citations")
    builder.add_edge("extract_citations", "agent")
    builder.add_edge("output_guard", END)

    return builder.compile()
```

**Flow diagram:**
```
START → input_guard → agent ──→ (has tool calls?) ──yes──→ tools → extract_citations → agent (loop)
                                       │
                                       no
                                       ↓
                                  output_guard → END
```

The agent can loop through tools multiple times (up to `agent_max_iterations * 4` recursion limit = 40 recursive calls).

### `run_agent()` — Public Interface:

```python
async def run_agent(message, session_id, history, active_sources=None) -> ChatResponse:
```
- Entry point called by `main.py`'s `/chat` endpoint.

1. **Builds initial state**: Combines conversation history + new user message.
2. **Runs the graph**: `await graph.ainvoke(initial_state, config=config)`.
3. **Extracts answer**: Finds the last `AIMessage` in the final state.
4. **Deduplicates citations**: Removes duplicate chunk_ids, sanitizes page numbers.
5. **Reconstructs tool calls**: Collects all tool invocations from messages for the response.
6. **Follow-up generation**:
   - Detects greetings → returns hardcoded suggestions like "What documents are indexed?"
   - For real queries → uses a separate LLM call with a focused prompt to generate 3 follow-up questions grounded in the retrieved chunks.
   - Uses `asyncio.wait_for()` with a **4-second timeout** to prevent blocking.

7. **Returns `ChatResponse`** with all fields populated.

---

## 16. FastAPI Application — `backend/main.py`

**Purpose:** The HTTP API layer. Defines all endpoints, middleware, file handling, and request processing.

### Rate Limiting:

```python
_RATE_LIMIT_WINDOW = 60     # seconds
_RATE_LIMIT_MAX_REQUESTS = 30
_rate_limit_store: Dict[str, List[float]] = {}
```
- Sliding window rate limiter: maximum 30 requests per 60-second window per IP address.

```python
def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    timestamps = _rate_limit_store.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        return False
    timestamps.append(now)
    _rate_limit_store[client_ip] = timestamps
    return True
```

### SSRF Protection:

```python
_BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
]
```
- Prevents Server-Side Request Forgery by blocking URLs that resolve to private/internal IP addresses.

```python
def _is_url_safe(url: str) -> bool:
    hostname = urlparse(url).hostname
    ip = ipaddress.ip_address(socket.gethostbyname(hostname))
    return not any(ip in network for network in _BLOCKED_IP_RANGES)
```

### Lifespan (Startup/Shutdown):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("application_startup", ...)
    get_vector_store()  # warm up ChromaDB connection
    yield
    logger.info("application_shutdown")
```
- Initializes logging and warms up the vector store connection on startup.

### Middleware:

```python
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("request_completed", method=request.method, path=request.url.path, status=response.status_code, duration_ms=round(duration * 1000))
    response.headers["X-Request-ID"] = request_id
    return response
```
- Assigns a UUID to each request, sets it in the logging context, measures duration, and returns it in the `X-Request-ID` header.

### Endpoint: `POST /chat`

```python
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
```
1. **Rate limit check**: Returns 429 if exceeded.
2. **Session management**: Creates new session ID if not provided.
3. **Load history**: Gets conversation history from `SessionStore`.
4. **Run agent**: Calls `run_agent()` with the message, session ID, history, and active sources.
5. **Save messages**: Stores the user message and AI response in the session store.
6. **Return response**: Returns the full `ChatResponse`.

### Endpoint: `POST /upload`

```python
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile, request: Request):
```
1. **Validates** file size (max 50MB) and extension.
2. **Saves** file to `uploads/` directory.
3. **Calls** `ingest_file()` to process through the full pipeline.
4. **Invalidates** source cache so the agent sees the new document.
5. **Returns** chunk count and source ID.

### Endpoint: `POST /ingest`

```python
@app.post("/ingest", response_model=UploadResponse)
async def ingest_url(req: IngestURLRequest, request: Request):
```
1. **URL validation**: Checks protocol (http/https only) and SSRF safety.
2. **Downloads** the file with a 30-second timeout and 50MB size limit.
3. **Detects** file type from Content-Type header or URL path.
4. **Saves** to `uploads/` and ingests.

### Endpoint: `POST /scrape`

```python
@app.post("/scrape", response_model=UploadResponse)
async def scrape_page(req: ScrapeRequest, request: Request):
```
1. **Fetches** the webpage HTML with a browser-like User-Agent.
2. **Parses** with BeautifulSoup (`lxml` parser).
3. **Removes noise**: Deletes `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<form>` tags.
4. **Extracts text** and removes lines that are subsets of other lines (set-based deduplication).
5. **Saves** as a `.txt` file and ingests.

### Endpoint: `GET /sources`

```python
@app.get("/sources")
async def list_sources():
```
- Lists all unique source documents in ChromaDB with their chunk counts and file types.

### Endpoint: `DELETE /sources/{filename}`

```python
@app.delete("/sources/{filename}")
async def delete_source(filename: str):
```
- Deletes all chunks for a source from ChromaDB and its SQL tables (if any).
- Invalidates the source cache.

### Session Endpoints:

- `GET /sessions` — Lists all active session IDs.
- `GET /sessions/{session_id}` — Returns full conversation history as `SessionHistory`.
- `DELETE /sessions/{session_id}` — Clears a session's history.

### Endpoint: `GET /`

```python
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse(str(FRONTEND_DIR / "chat.html"))
```
- Serves the frontend chat UI.

---

## 17. Frontend

### `frontend/chat.html`
- Full-featured chat interface with:
  - Message input with send button
  - File upload with drag-and-drop
  - URL ingestion form
  - Web scraping form
  - Source document management panel
  - Citation display with expandable snippets
  - Follow-up suggestion chips
  - Session management
  - Responsive design

### `frontend/index.html`
- Redirects to `chat.html`.

---

## 18. Infrastructure

### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Lightweight Python 3.11 image. Installs dependencies, copies code, exposes port 8000.

### `docker-compose.yml`
- Defines the single service with port mapping, volume mounts for data persistence, and environment variable passthrough.

### `Makefile`
- Convenience targets: `make run`, `make docker`, `make test`, etc.

### `run_dev.sh`
- Shell script for local development that activates the virtual environment and starts uvicorn with auto-reload.

---

## 19. Data Flow Summary

### Ingestion Flow:
```
User uploads file → FastAPI /upload endpoint
  → Validate (size, type)
  → Save to uploads/
  → indexer.ingest_file()
    → Delete existing chunks (dedup)
    → document_loader.load_file()    → [(text, metadata), ...]
    → chunker.chunk_documents()      → [{chunk_id, text, metadata}, ...]
    → embedder.embed()               → [[0.1, 0.3, ...], ...]
    → vector_store.add()             → ChromaDB persistent storage
    → _load_into_sql()              → SQLite (CSV/JSON only)
  → Invalidate source cache
  → Return chunk count
```

### Query Flow:
```
User sends message → FastAPI /chat endpoint
  → Rate limit check
  → Session history lookup
  → run_agent(message, session_id, history, active_sources)
    → LangGraph state machine:
      1. input_guard_node:
         - Query rewriting (non-English/typos)
         - PII redaction
         - Injection detection
      2. agent_node:
         - Build dynamic prompt (sources + SQL tables)
         - Trim messages to token limit
         - LLM decides: use tools or answer directly
      3. tools (if needed):
         - Execute retrieve_documents / get_document_chunks / query_database / etc.
      4. extract_citations:
         - Parse tool results for citations
         - Detect clarification requests
      5. (Loop back to agent_node if more tools needed)
      6. output_guard_node:
         - PII redaction on output
         - Filter low-score citations
         - Calculate confidence
         - Low confidence override
    → Generate follow-up suggestions (async, 4s timeout)
    → Build ChatResponse
  → Save messages to session store
  → Return response with answer, citations, tool_calls, follow_ups, confidence
```
