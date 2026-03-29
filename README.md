# Agentic RAG Chatbot

A production-style, locally-runnable Retrieval-Augmented Generation (RAG) chatbot with a full agentic workflow using **LangGraph**, **ChromaDB**, **SQLite**, and a **FastAPI** backend.

## Architecture Overview
```
User ──► FastAPI /chat
              │
              ▼
       Input Guardrails
       (PII redact, injection detect)
              │
              ▼
        LangGraph Agent  ◄──── Conversation Memory (session-scoped)
         (ReAct loop)
        /      |      \
       ▼       ▼       ▼
  Vector    SQL DB   Clarify
  Search    Query    Tool
  (Chroma)  (SQLite)
        \      |      /
         ▼     ▼     ▼
       Output Synthesizer
       (with citations)
              │
              ▼
       Output Guardrails
       (confidence check)
              │
              ▼
           Response
```

## Live Demo

**Try it now:** (https://agentic-rag-chatbot-db4q.onrender.com/)

> No setup needed — just enter your OpenAI API key in the UI and start chatting.

## Quick Start

### Option A — Docker (Recommended)
```bash
cp .env.example .env          # leave OPENAI_API_KEY blank (users provide via UI)
make docker-up                # builds + starts everything
make seed                     # seeds the SQLite DB with sample data
open http://localhost:8000    # chat UI
```

### Option B — Local (virtualenv)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make seed
make dev
```

## Supported File Types

| Format | Ingestion Method | Notes |
|--------|-----------------|-------|
| PDF    | pypdf (page-aware) | Preserves page numbers in citations |
| TXT    | Plain text read | Chunked by sentence |
| DOCX   | python-docx | Paragraph-level chunking |
| JSON   | Key-value flattening | Each top-level key → chunk |
| CSV    | Row-by-row text | Column headers preserved |
| SQL table | SQLAlchemy introspection | Schema + data indexed |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message, get agentic response |
| `/upload` | POST | Upload a file for ingestion |
| `/ingest` | POST | Download & ingest a file from URL |
| `/scrape` | POST | Scrape a web page & ingest its text |
| `/sources` | GET | List indexed sources |
| `/sources/{filename}` | DELETE | Remove a document from the index |
| `/sources` | DELETE | Clear all knowledge base data |
| `/sessions/{id}` | GET | Retrieve conversation history |
| `/sessions/{id}` | DELETE | Clear a session |
| `/sessions` | GET | List all active sessions |
| `/health` | GET | Health check |

## Environment Variables

See `.env.example` for full list. Key variables:
- `OPENAI_API_KEY` — Optional server-side fallback key (users provide their own via the UI)
- `MOCK_MODE` — `true` bypasses all LLM/embedding calls
- `OPENAI_BASE_URL` — Override for any OpenAI-compatible API (Ollama, LM Studio, etc.)
- `ENABLE_PII_REDACTION` — Redact PII from user input before processing

## Mock Mode (No API Key Needed)

Set `MOCK_MODE=true` in `.env`. The system uses:
- Deterministic keyword-based "LLM" responses
- Random unit-norm vectors as embeddings
- Full pipeline still runs (ChromaDB, SQLite, guardrails, memory)

This lets you test the entire system without an OpenAI key.

## Deployment (Render)

This project is deployed on [Render](https://render.com) for 24/7 availability:

1. Push to GitHub
2. Connect the repo to Render as a Docker web service
3. Leave `OPENAI_API_KEY` blank — users provide their own key via the UI
4. The `render.yaml` blueprint auto-configures all environment variables

See `render.yaml` for the full deployment configuration.

## Per-User API Key

Users provide their own OpenAI API key through a modal in the UI. Keys are:
- Stored only in the browser's `localStorage` (never sent to any server-side storage)
- Sent per-request in the POST body to `/chat`
- Never logged or persisted on the server

This means no server-side API key is needed for the app to function.

## Local LLM via Ollama
```bash
ollama pull llama3
# In .env:
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
```
