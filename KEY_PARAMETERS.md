# Agentic RAG Chatbot — Key Parameters & Critical Code Reference

This document highlights the most important parameters, configuration values, and critical lines of code that control the behavior, performance, and security of the system. Use this as a quick reference for tuning, debugging, and understanding system behavior.

---

## Table of Contents

1. [Configuration Parameters](#1-configuration-parameters)
2. [Chunking & Embedding Parameters](#2-chunking--embedding-parameters)
3. [Retrieval & Ranking Parameters](#3-retrieval--ranking-parameters)
4. [Agent & LLM Parameters](#4-agent--llm-parameters)
5. [Token Management Parameters](#5-token-management-parameters)
6. [Caching Parameters](#6-caching-parameters)
7. [Rate Limiting & Security Parameters](#7-rate-limiting--security-parameters)
8. [Guardrail Thresholds](#8-guardrail-thresholds)
9. [Session & Memory Parameters](#9-session--memory-parameters)
10. [File Processing Limits](#10-file-processing-limits)
11. [Critical Code Patterns](#11-critical-code-patterns)
12. [Performance Tuning Guide](#12-performance-tuning-guide)

---

## 1. Configuration Parameters

All settings are defined in `backend/config.py` and loaded from the `.env` file.

| Parameter | Default | File & Line | Purpose |
|-----------|---------|-------------|---------|
| `openai_api_key` | `""` | `config.py` Settings class | OpenAI API key. Empty = mock mode activates |
| `chroma_persist_dir` | `data/chroma_db/` | `config.py` Settings class | ChromaDB persistent storage directory |
| `upload_dir` | `uploads/` | `config.py` Settings class | Temporary file upload directory |
| `log_level` | `"INFO"` | `config.py` Settings class | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| `mock_mode` | `False` | `config.py` Settings class | Use mock LLM/embedder (no API calls) |
| `sql_db_path` | `data/structured.db` | `config.py` Settings class | SQLite database for CSV/JSON structured data |

**Critical code — Settings singleton:**
```python
# backend/config.py
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    ensure_dirs(s)
    return s
```
> The `@lru_cache(maxsize=1)` ensures only ONE Settings instance exists. This means `.env` changes require a server restart.

---

## 2. Chunking & Embedding Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `chunk_size` | **512** chars | `config.py` | Max characters per text chunk. Larger = more context per chunk but less precise retrieval |
| `chunk_overlap` | **64** chars | `config.py` | Overlap between adjacent chunks. Prevents information loss at boundaries |
| `embedding_model` | `text-embedding-3-small` | `config.py` | OpenAI embedding model. Determines vector quality and cost |
| `embedding_dimensions` | **1536** | `config.py` | Vector dimensionality. Must match the model's output |
| `_BATCH_SIZE` | **100** | `embedder.py` RealEmbedder | Texts per API call. Prevents rate limit errors |
| `MIN_CHUNK_LENGTH` | **20** chars | `document_loader.py` | Minimum paragraph length before merging with neighbors |

**Critical code — Chunk splitting:**
```python
# backend/ingestion/chunker.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,        # 512
    chunk_overlap=chunk_overlap,  # 64
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # Priority: paragraph > line > sentence > word > char
)
```
> The separator hierarchy is key to chunk quality. It tries paragraph breaks first, preserving semantic boundaries.

**Critical code — Context prefix:**
```python
# backend/ingestion/chunker.py
def _build_context_prefix(metadata: Dict) -> str:
    # Produces: "[Document: report.pdf | Type: pdf | Page: 3] "
    parts = []
    if metadata.get("source"):
        parts.append(f"Document: {metadata['source']}")
    ...
    return f"[{' | '.join(parts)}] " if parts else ""
```
> Every chunk is prefixed with source metadata. This helps both the embedding model and the LLM understand the chunk's origin.

**Critical code — Chunk ID format:**
```python
# backend/ingestion/chunker.py
chunk_id = f"{metadata.get('source', 'unknown')}::chunk_{global_idx}"
# Example: "report.pdf::chunk_42"
```
> Chunk IDs are deterministic: same file, same order = same IDs. Used for deduplication and citation tracking.

---

## 3. Retrieval & Ranking Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `retrieval_top_k` | **10** | `config.py` | Number of chunks retrieved from ChromaDB per search |
| `k_constant` | **60** | `tools.py` `_reciprocal_rank_fusion()` | RRF smoothing constant. Higher = semantic ranking dominates |
| `confidence_threshold` | **0.25** | `config.py` | Minimum confidence to trust an answer |
| Mismatch threshold | **0.60** (60%) | `tools.py` `retrieve_documents` | Keyword overlap ratio below which a warning is added |
| Mismatch score gate | **0.35** | `tools.py` `retrieve_documents` | Only warn if BOTH overlap < 60% AND score < 0.35 |
| Citation noise floor | **0.10** | `graph.py` `output_guard_node` | Citations below this score are filtered out |
| `MAX_SNIPPET_CHARS` | **1200** | `tools.py` `get_document_chunks` | Max characters per chunk snippet in full-doc retrieval |
| `max_chunks` | **150** | `tools.py` `get_document_chunks` | Max chunks returned for document summarization |
| Chunk cap | **300** | `tools.py` `get_document_chunks` | Hard limit on ChromaDB query |
| SQL row limit | **50** | `tools.py` `query_database` | Max rows returned by SQL queries |
| Vector store batch | **5000** | `vector_store.py` `add()` / `delete_by_source()` | ChromaDB operations batched per 5000 items |

**Critical code — Reciprocal Rank Fusion:**
```python
# backend/agents/tools.py
def _reciprocal_rank_fusion(semantic_results, query, k_constant=60):
    ...
    rrf_score = (1.0 / (k_constant + s_rank)) + (1.0 / (k_constant + k_rank))
```
> RRF combines semantic similarity rank and keyword overlap rank. `k_constant=60` means a document must be ranked high on BOTH signals to get the top score. Lower k_constant makes ranking differences more dramatic.

**Critical code — Confidence calculation:**
```python
# backend/agents/graph.py — output_guard_node
scores = [c.get("score", 0) for c in cits]
avg_score = sum(scores) / len(scores)
max_score = max(scores)
confidence = round(0.6 * max_score + 0.4 * avg_score, 3)
```
> The 60/40 weighted formula prioritizes the **best matching chunk** over the average. This means one highly relevant chunk can produce a confident answer even if other chunks are weak.

**Critical code — Query mismatch detection:**
```python
# backend/agents/tools.py — retrieve_documents
query_words = set(re.findall(r'\w{3,}', query.lower())) - _STOP_WORDS
...
match_ratio = len(overlap) / max(len(query_words), 1)
if match_ratio < 0.60 and r["score"] < 0.35:
    r["mismatch_warning"] = "Low keyword overlap with query"
```
> Double-gated warning: requires BOTH low keyword overlap AND low semantic score. Prevents false positives on paraphrased but semantically relevant results.

---

## 4. Agent & LLM Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `llm_model` | `gpt-4o-mini` | `config.py` | The reasoning model. Balances speed/cost/quality |
| `temperature` | **0** | `graph.py` `_build_llm()` | Deterministic outputs (no randomness) |
| `max_retries` | **2** | `graph.py` `_build_llm()` | Auto-retry on transient API failures |
| `request_timeout` | **30** seconds | `graph.py` `_build_llm()` | Per-request timeout to the OpenAI API |
| `agent_max_iterations` | **10** | `config.py` | Max reasoning loops (LLM→tools→LLM) |
| `recursion_limit` | **80** (or `iterations × 4`) | `graph.py` `run_agent()` | LangGraph recursion limit |
| Follow-up timeout | **4.0** seconds | `graph.py` `run_agent()` | Async timeout for follow-up suggestion generation |
| Follow-up count | **3** | `graph.py` `run_agent()` | Number of follow-up suggestions generated |
| Follow-up char limit | **55** chars | `graph.py` `run_agent()` | Max characters per follow-up question |

**Critical code — LLM configuration:**
```python
# backend/agents/graph.py
return ChatOpenAI(
    model=settings.llm_model,      # "gpt-4o-mini"
    temperature=0,                  # Deterministic
    max_retries=2,                  # Auto retry
    request_timeout=30,             # 30s timeout
)
```

**Critical code — Recursion limit:**
```python
# backend/agents/graph.py — run_agent()
config = RunnableConfig(
    recursion_limit=max(80, get_settings().agent_max_iterations * 4),
    configurable={"session_id": session_id},
)
```
> The recursion limit must be at least 4× the max iterations because each iteration involves multiple graph nodes (agent → tools → extract_citations → agent).

**Critical code — Follow-up generation with timeout:**
```python
# backend/agents/graph.py — run_agent()
followup_resp = await asyncio.wait_for(
    followup_llm.ainvoke(_followup_messages),
    timeout=4.0,  # Don't block the response for slow follow-up generation
)
```
> The 4-second timeout ensures follow-up suggestions don't delay the main response. If the LLM takes too long, follow-ups are simply omitted.

---

## 5. Token Management Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `_MAX_HISTORY_TOKENS` | **60,000** | `graph.py` | Max tokens for conversation history |
| `_SYSTEM_PROMPT_BUFFER` | **4,000** | `graph.py` | Reserved tokens for system prompt |
| `_RESPONSE_BUFFER` | **4,000** | `graph.py` | Reserved tokens for LLM response |
| Effective history budget | ~52,000 tokens | Computed | `60000 - 4000 - 4000` |
| Token fallback estimate | **4 chars/token** | `graph.py` `_count_tokens()` | Fallback when tiktoken fails |

**Critical code — Token counting:**
```python
# backend/agents/graph.py
def _count_tokens(text: str) -> int:
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4  # ~4 chars per token as rough estimate
```

**Critical code — Message trimming with atomic blocks:**
```python
# backend/agents/graph.py
def _trim_messages_to_token_limit(messages):
    budget = _MAX_HISTORY_TOKENS - _SYSTEM_PROMPT_BUFFER - _RESPONSE_BUFFER
    ...
    # Keep tool_call + ToolMessage pairs together
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        block = [msg]
        while j < len(messages) and isinstance(messages[j], ToolMessage):
            block.append(messages[j])
            j += 1
```
> Messages with tool calls are kept as atomic blocks with their ToolMessage responses. Splitting them would corrupt the conversation history.

---

## 6. Caching Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| Source cache TTL | **30.0** seconds | `graph.py` `_SOURCE_CACHE_TTL` | How long cached source names are valid |
| Settings cache | **1** entry | `config.py` `@lru_cache(maxsize=1)` | Singleton settings — never expires |
| Vector store cache | **1** entry | `vector_store.py` `@lru_cache(maxsize=1)` | Singleton ChromaDB connection — never expires |
| SQL store cache | **1** entry | `sql_store.py` `get_sql_store()` | Singleton SQLite connection |
| Session store cache | **1** entry | `session_store.py` global singleton | In-memory session storage |
| Tokenizer cache | Lazy singleton | `graph.py` `_get_tokenizer()` | tiktoken tokenizer loaded once |
| Graph cache | Lazy singleton | `graph.py` `get_agent_graph()` | Compiled LangGraph loaded once |

**Critical code — Source cache invalidation:**
```python
# backend/agents/graph.py
def invalidate_source_cache():
    global _source_cache_names
    _source_cache_names = None
```
> **Must be called** after every upload, ingestion, scrape, or deletion. Otherwise the agent won't see new/removed documents for up to 30 seconds.

**Critical code — Vector store singleton (potential pitfall):**
```python
# backend/retrieval/vector_store.py
@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStoreAdapter:
    settings = get_settings()
    return ChromaVectorStoreAdapter(persist_dir=settings.chroma_persist_dir, ...)
```
> Because `@lru_cache` never expires, if the ChromaDB files are deleted while the server is running, the cached connection becomes stale and throws "readonly database" errors. **Fix: restart the server** after deleting data files.

---

## 7. Rate Limiting & Security Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| Rate limit window | **60** seconds | `main.py` `_RATE_LIMIT_WINDOW` | Sliding window duration |
| Max requests | **30** per window | `main.py` `_RATE_LIMIT_MAX_REQUESTS` | Requests allowed per IP per window |
| Max file size | **50** MB | `config.py` `max_file_size_mb` | Upload file size limit |
| URL download timeout | **30** seconds | `main.py` `/ingest` endpoint | Timeout for downloading remote files |
| URL max download size | **50** MB | `main.py` `/ingest` endpoint | Size limit for URL downloads |

**Critical code — SSRF protection:**
```python
# backend/main.py
_BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),     # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),    # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),   # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
]
```
> Prevents Server-Side Request Forgery by blocking URLs that resolve to internal network addresses. Applied to both `/ingest` and `/scrape` endpoints.

**Critical code — SQL injection prevention:**
```python
# backend/retrieval/sql_store.py
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
```
> Blocks all write/admin SQL operations. Combined with a `SELECT`-only prefix check for defense in depth.

---

## 8. Guardrail Thresholds

### Injection Detection Rules

| Rule | Pattern | Risk Score | Threshold |
|------|---------|------------|-----------|
| `ignore_previous` | `ignore (all)? (previous\|above\|prior) (instructions\|prompts\|rules)` | **0.9** | 0.5 |
| `role_switch` | `you are now (a\|an\|the)? \w+` | **0.8** | 0.5 |
| `jailbreak_dan` | `DAN \| do anything now \| jailbreak` | **0.95** | 0.5 |
| `system_prompt_leak` | `(show\|reveal\|print\|output) (your)? (system prompt\|instructions\|rules)` | **0.85** | 0.5 |
| `delimiter_injection` | ` ```(system\|assistant\|instruction) ` | **0.6** | 0.5 |
| `base64_payload` | 40+ base64 chars | **0.4** | 0.5 (won't trigger alone) |
| `command_execution` | `(run\|execute\|eval\|exec)\(` | **0.9** | 0.5 |

```python
# backend/guardrails/injection_detector.py
_INJECTION_THRESHOLD = 0.5
```
> A single rule with risk ≥ 0.5 is enough to flag the input as an injection. The `base64_payload` rule at 0.4 requires an additional rule to trigger detection.

### PII Detection Patterns

| Type | Pattern | Example Match |
|------|---------|---------------|
| `EMAIL` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` | `user@example.com` |
| `PHONE_US` | `\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` | `(555) 123-4567` |
| `SSN` | `\b\d{3}-\d{2}-\d{4}\b` | `123-45-6789` |
| `CREDIT_CARD` | `\b(?:\d{4}[-\s]?){3}\d{4}\b` | `4111-1111-1111-1111` |
| `IP_ADDRESS` | `\b(?:\d{1,3}\.){3}\d{1,3}\b` | `192.168.1.1` |
| `DATE_OF_BIRTH` | `\b(?:0[1-9]\|1[0-2])[/-](?:0[1-9]\|[12]\d\|3[01])[/-](?:19\|20)\d{2}\b` | `01/15/1990` |

**Critical code — Redaction strategy (reverse order):**
```python
# backend/guardrails/pii_redactor.py
for match in sorted(matches, key=lambda m: m.start, reverse=True):
    redacted = redacted[:match.start] + f"[{match.entity_type}]" + redacted[match.end:]
```
> Processes matches from end to start so that replacing text doesn't shift the positions of earlier matches.

---

## 9. Session & Memory Parameters

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| `_MAX_HISTORY` | **20** messages | `session_store.py` | Max messages per session (FIFO) |
| Storage type | In-memory `Dict` | `session_store.py` | Lost on server restart |
| Thread safety | `threading.Lock` | `session_store.py` | One lock per store instance |
| Default confidence | **0.5** | `graph.py` `output_guard_node` | When no vector citations exist |
| Default confidence (initial) | **1.0** | `graph.py` `run_agent()` | Initial state before processing |

**Critical code — Session trimming:**
```python
# backend/memory/session_store.py
if len(self._sessions[session_id]) > _MAX_HISTORY:
    self._sessions[session_id] = self._sessions[session_id][-_MAX_HISTORY:]
```
> Keeps only the last 20 messages. Older messages are permanently discarded. This is independent of the token-based trimming in graph.py which operates on the messages passed to the LLM.

---

## 10. File Processing Limits

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| Supported extensions | `.pdf .txt .md .docx .json .csv` | `document_loader.py` | Allowed file types |
| JSON group size | **5** keys per group | `document_loader.py` `load_json()` | Groups dict keys for chunking |
| DOCX min paragraph | **20** chars | `document_loader.py` `MIN_CHUNK_LENGTH` | Merge threshold for short paragraphs |
| Source ID length | **12** hex chars | `indexer.py` `_make_source_id()` | SHA1 truncation for source IDs |
| ChromaDB batch size | **5000** | `vector_store.py` | Items per add/delete operation |
| Embedding batch size | **100** | `embedder.py` `_BATCH_SIZE` | Texts per OpenAI embedding API call |

**Critical code — Deduplication on re-upload:**
```python
# backend/ingestion/indexer.py — ingest_file()
store = get_vector_store()
store.delete_by_source(file_name)  # Delete ALL existing chunks for this file
```
> Every upload first deletes all existing chunks for that filename. This is the deduplication mechanism — re-uploading replaces rather than duplicates.

---

## 11. Critical Code Patterns

### Pattern 1: Singleton with `@lru_cache`
```python
@lru_cache(maxsize=1)
def get_settings() -> Settings: ...
def get_vector_store() -> ChromaVectorStoreAdapter: ...
```
> Used for Settings, VectorStore, SQLStore, SessionStore, Graph, and Tokenizer. Ensures one instance across the entire application.

### Pattern 2: Lazy Global Initialization
```python
_graph = None
def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
```
> The graph is compiled only on first use, not at import time. This avoids circular imports and startup delays.

### Pattern 3: Request-Scoped Logging
```python
_request_id: ContextVar[str] = ContextVar("request_id", default="no-request")
```
> Every HTTP request gets a UUID, stored in a `ContextVar`. All log entries within that request automatically include the ID, making it easy to trace a single request through the entire pipeline.

### Pattern 4: Defense in Depth (SQL)
```python
# Layer 1: Regex blocks forbidden keywords
if _FORBIDDEN_SQL.search(sql_stripped): raise ValueError(...)
# Layer 2: Must start with SELECT
if not sql_stripped.upper().startswith("SELECT"): raise ValueError(...)
```
> Two independent checks for SQL safety. Even if one is bypassed, the other catches it.

### Pattern 5: Atomic Tool Call Blocks
```python
# When trimming messages, keep tool_call + ToolMessage together
if hasattr(msg, "tool_calls") and msg.tool_calls:
    block = [msg]
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        block.append(messages[j])
        j += 1
```
> Splitting an AIMessage with tool_calls from its ToolMessage responses would corrupt the conversation structure and cause LLM errors.

### Pattern 6: Graceful Degradation
```python
# Embedder: try real, fall back to mock
try:
    return RealEmbedder(...)
except Exception:
    return MockEmbedder(...)
```
> The system remains functional even without an API key by falling back to mock implementations.

### Pattern 7: Set-Based Web Scrape Deduplication
```python
# backend/main.py — /scrape endpoint
unique_lines = []
line_set = set()
for line in lines:
    normalized = line.strip().lower()
    if normalized not in line_set:
        is_subset = any(normalized in existing for existing in line_set if len(existing) > len(normalized))
        if not is_subset:
            unique_lines.append(line)
            line_set.add(normalized)
```
> Removes duplicate lines AND lines that are substrings of other lines. This eliminates navigation/menu text that appears in multiple places.

---

## 12. Performance Tuning Guide

### To improve answer quality:
| Change | Parameter | Current → Suggested |
|--------|-----------|-------------------|
| More context per chunk | `chunk_size` | 512 → 768 or 1024 |
| More retrieval results | `retrieval_top_k` | 10 → 15 or 20 |
| Better model | `llm_model` | `gpt-4o-mini` → `gpt-4o` |
| Lower confidence gate | `confidence_threshold` | 0.25 → 0.15 |

### To improve speed:
| Change | Parameter | Current → Suggested |
|--------|-----------|-------------------|
| Fewer chunks retrieved | `retrieval_top_k` | 10 → 5 |
| Shorter history | `_MAX_HISTORY_TOKENS` | 60000 → 30000 |
| Faster model | `llm_model` | keep `gpt-4o-mini` |
| Skip follow-ups | Follow-up timeout | 4.0 → 1.0 |

### To reduce costs:
| Change | Parameter | Current → Suggested |
|--------|-----------|-------------------|
| Smaller chunks (fewer embeddings) | `chunk_size` | 512 → 1024 |
| Fewer embedding dimensions | `embedding_dimensions` | 1536 → 512 (with `text-embedding-3-small`) |
| Fewer retrieval results | `retrieval_top_k` | 10 → 5 |
| Shorter conversation context | `_MAX_HISTORY_TOKENS` | 60000 → 20000 |
| Limit agent iterations | `agent_max_iterations` | 10 → 5 |

### To improve security:
| Change | Parameter | Current → Suggested |
|--------|-----------|-------------------|
| Stricter injection | `_INJECTION_THRESHOLD` | 0.5 → 0.3 |
| Enable Presidio | `PIIRedactor(use_presidio=True)` | False → True |
| Tighter rate limits | `_RATE_LIMIT_MAX_REQUESTS` | 30 → 10 |
| Smaller file uploads | `max_file_size_mb` | 50 → 10 |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│              MOST IMPORTANT PARAMETERS              │
├─────────────────────────────────────────────────────┤
│ chunk_size           = 512 chars                    │
│ chunk_overlap        = 64 chars                     │
│ retrieval_top_k      = 10 results                   │
│ confidence_threshold = 0.25                         │
│ k_constant (RRF)     = 60                           │
│ _MAX_HISTORY_TOKENS  = 60,000                       │
│ _MAX_HISTORY         = 20 messages                  │
│ _SOURCE_CACHE_TTL    = 30 seconds                   │
│ temperature          = 0 (deterministic)            │
│ max_file_size_mb     = 50 MB                        │
│ agent_max_iterations = 10                           │
│ follow-up timeout    = 4.0 seconds                  │
│ rate limit           = 30 req / 60 sec              │
│ injection threshold  = 0.5 risk score               │
│ citation noise floor = 0.10 score                   │
│ confidence formula   = 0.6×max + 0.4×avg            │
└─────────────────────────────────────────────────────┘
```
