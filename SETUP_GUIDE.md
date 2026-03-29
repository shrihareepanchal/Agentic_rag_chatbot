# Agentic RAGChat — Setup & Run Guide

This is a complete AI chatbot project that can read your documents (PDFs, Word files, CSVs, etc.) and answer questions about them using OpenAI's GPT-4o-mini.

**Live version:** [https://naresh-ragchat.onrender.com](https://naresh-ragchat.onrender.com) — no setup needed, just provide your OpenAI API key in the UI.

---

## Prerequisites (What You Need First)

Before we begin, make sure you have these installed on your computer:

### 1. Python 3.11 or newer

**Check if you have it:**
Open your terminal and type:
```bash
python3 --version
```
You should see something like `Python 3.11.x` or `Python 3.12.x`.

**If you don't have it:**
- **Mac**: Download from https://www.python.org/downloads/ or use Homebrew:
  ```bash
  brew install python@3.11
  ```

### 2. VS Code (Visual Studio Code)

Download from: https://code.visualstudio.com/

### 3. OpenAI API Key

You need an API key to use GPT-4o-mini (the AI brain of this project).

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-...`) — you'll enter it in the chat UI when prompted

> **Note**: OpenAI charges for API usage. GPT-4o-mini is very affordable (~$0.15 per 1M input tokens). A typical chat session costs less than $0.01.
> 
> **New**: You no longer need to put your API key in a `.env` file. The chat UI will prompt you to enter your key, and it’s stored securely in your browser only.

---

## Step-by-Step Setup

### Step 1: Unzip the Project

Unzip `agentic-rag.zip` to a folder on your computer. For example:
- **Mac/Linux**: `~/Desktop/agentic-rag`
- **Windows**: `C:\Users\YourName\Desktop\agentic-rag`

### Step 2: Open in VS Code

1. Open VS Code
2. Go to **File → Open Folder** (or `Cmd+O` on Mac / `Ctrl+K Ctrl+O` on Windows)
3. Navigate to the `agentic-rag` folder and open it

You should see the project structure in the left sidebar:
```
agentic-rag/
├── backend/
├── frontend/
├── data/
├── .env
├── requirements.txt
├── Dockerfile
└── ...
```

### Step 3: Open the Terminal in VS Code

1. Go to **Terminal → New Terminal** (or press `` Ctrl+` ``)
2. A terminal panel opens at the bottom of VS Code
3. Make sure you're inside the `agentic-rag` folder:
   ```bash
   pwd
   ```
   Should show something like `/Users/yourname/Desktop/agentic-rag`

### Step 4: Create a Virtual Environment

A virtual environment is an isolated Python setup so this project's packages don't interfere with other projects on your computer.

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **How to know it worked**: You should see `(.venv)` at the beginning of your terminal prompt, like:
> ```
> (.venv) yourname@computer agentic-rag %
> ```

### Step 5: Set Your OpenAI API Key

The chatbot now has a built-in API key input in the UI. When you open the chat interface, click the **"Set API Key"** button in the top-right corner and enter your key.

Alternatively, you can still set a server-side fallback key in the `.env` file:
1. In VS Code's file explorer (left sidebar), find and click on the `.env` file
2. Find this line:
   ```
   OPENAI_API_KEY=
   ```
3. Add your key (optional — users can provide their own via the UI):
   ```
   OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
   ```
4. Save the file (`Cmd+S` on Mac / `Ctrl+S` on Windows)

> **IMPORTANT**: Never share your API key publicly or commit it to GitHub.

### Step 6: Install Python Dependencies

In the VS Code terminal (make sure `.venv` is activated), run:

```bash
pip install -r requirements.txt
```

This installs all the Python packages the project needs. It may take 2–5 minutes depending on your internet speed.

**What you should see**: A lot of text scrolling by as packages download. It should end with something like:
```
Successfully installed fastapi-0.111.0 langchain-0.2.6 chromadb-0.5.3 ...
```

**If you see errors**:
- **"pip not found"**: Try `pip3 install -r requirements.txt`
- **Build errors on Mac**: Run `xcode-select --install` first, then retry
- **Build errors on Windows**: Install Visual Studio Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Step 7: Install the spaCy Language Model (Optional)

This is needed only if you enable PII (personal information) detection:

```bash
python -m spacy download en_core_web_sm
```

If this fails, that's OK — the project works without it. PII detection will fall back to regex mode.

### Step 8: Start the Backend Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**What you should see**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> **Tip**: If you want the server to auto-reload when you change code (useful during development), add `--reload`:
> ```bash
> uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
> ```

### Step 9: Open the Chat Interface

Open your web browser and go to:

```
http://localhost:8000
```

You should see the **Agentic RAGChat** interface with a welcome screen!

---

## Using the Chatbot

### Upload a File

1. Click the **📎 Attach** button at the bottom of the chat
2. Choose **"Upload File"**
3. Select a file from your computer (PDF, DOCX, TXT, CSV, JSON, or XLSX)
4. Wait for the success message (e.g., "Successfully indexed 24 chunks from report.pdf")

### Clear Knowledge Base

To remove all uploaded documents and indexed data, click the **"🗑 Clear All"** button in the Knowledge Base sidebar. This deletes all vectors, SQL tables, and uploaded files.

### Ask Questions

Type your question in the chat box and press Enter or click Send. Examples:
- "Summarize the uploaded document"
- "What are the key findings?"
- "How many records are in the CSV file?"
- "What does page 3 say?"

### Try Different Input Methods

- **Upload File**: Click 📎 → Upload File → select from your computer
- **Import from URL**: Click 📎 → Import from URL → paste a direct link to a PDF/DOCX/CSV
- **Scrape Web Page**: Click 📎 → Scrape Web Page → paste any website URL to extract its text

---

## Using the Standalone Chat Page (Optional)

There's also a standalone version that can connect to the backend from anywhere:

1. Open a **second terminal** in VS Code (click the `+` icon in the terminal panel)
2. Run:
   ```bash
   cd frontend && python3 -m http.server 3000
   ```
3. Open your browser to:
   ```
   http://localhost:3000/chat.html?api=http://localhost:8000
   ```

This version remembers your API URL across browser sessions.

---

## Recommended VS Code Extensions

Install these extensions for a better experience (click the Extensions icon in the left sidebar or press `Cmd+Shift+X` / `Ctrl+Shift+X`):

| Extension | Why |
|-----------|-----|
| **Python** (by Microsoft) | Python language support, IntelliSense, debugging |
| **Pylance** (by Microsoft) | Advanced Python type checking and autocomplete |
| **Python Debugger** (by Microsoft) | Step-through debugging for the backend |
| **Even Better TOML** | Syntax highlighting for config files |
| **Docker** (by Microsoft) | If you want to use Docker deployment |
| **REST Client** | Test API endpoints directly from VS Code |

---

## Project Structure (Quick Reference)

```
agentic-rag/
│
├── .env                     ← Your API key and settings go here
├── requirements.txt         ← Python package list
│
├── backend/                 ← Server-side code
│   ├── main.py              ← API server (start point)
│   ├── config.py            ← Configuration loader
│   ├── agents/
│   │   ├── graph.py         ← AI agent pipeline (the brain)
│   │   └── tools.py         ← Tools the AI can use
│   ├── ingestion/
│   │   ├── document_loader.py  ← Read files
│   │   ├── chunker.py       ← Split text into pieces
│   │   ├── embedder.py      ← Convert text to vectors
│   │   └── indexer.py       ← Full ingestion pipeline
│   ├── retrieval/
│   │   ├── vector_store.py  ← ChromaDB (semantic search)
│   │   └── sql_store.py     ← SQLite (structured queries)
│   ├── guardrails/
│   │   ├── injection_detector.py  ← Block hacking attempts
│   │   └── pii_redactor.py        ← Hide personal info
│   ├── memory/
│   │   └── session_store.py ← Chat history
│   ├── models/
│   │   └── schemas.py       ← Data models
│   └── utils/
│       └── logger.py        ← Logging
│
├── frontend/
│   ├── index.html           ← Main chat UI
│   └── chat.html            ← Standalone version
│
├── data/                    ← Databases (auto-created)
├── uploads/                 ← Uploaded files (auto-created)
└── logs/                    ← Log files (auto-created)
```

---

## API Endpoints (For Developers)

Once the server is running, you can also access the auto-generated API docs at:

```
http://localhost:8000/docs
```

This shows an interactive Swagger UI where you can test every endpoint.

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/` | GET | Serves the chat UI |
| `/health` | GET | Server status check |
| `/chat` | POST | Send a message, get AI response |
| `/upload` | POST | Upload a file for indexing |
| `/ingest` | POST | Download & index a file from URL |
| `/scrape` | POST | Scrape a web page & index it |
| `/sources` | GET | List all indexed documents |
| `/sources/{filename}` | DELETE | Remove a document |
| `/sources` | DELETE | Clear all knowledge base data |
| `/sessions/{id}` | GET | Get chat history |
| `/sessions/{id}` | DELETE | Clear a chat session |
| `/sessions` | GET | List all active sessions |

---

## Common Issues & Fixes

### "ModuleNotFoundError: No module named 'backend'"
**Cause**: You're not in the project root directory.
**Fix**: Make sure your terminal is in the `agentic-rag` folder:
```bash
cd /path/to/agentic-rag
```

### "Address already in use" (port 8000)
**Cause**: Something else is already running on port 8000.
**Fix**:
```bash
# Mac/Linux: Kill whatever is on port 8000
lsof -ti:8000 | xargs kill -9

# Then restart the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### "openai.AuthenticationError" or "Invalid API Key"
**Cause**: Your OpenAI API key is wrong or expired.
**Fix**: 
1. Go to https://platform.openai.com/api-keys
2. Create a new key
3. Update it in the `.env` file
4. Restart the server

### "No module named 'chromadb'" or similar
**Cause**: Dependencies not installed properly.
**Fix**:
```bash
# Make sure virtual environment is active
source .venv/bin/activate   # Mac/Linux
# or
.venv\Scripts\activate      # Windows

# Reinstall everything
pip install -r requirements.txt
```

### Server starts but chat gives no response
**Cause**: Likely a network or API issue.
**Fix**: 
1. Check the terminal for error messages
2. Visit `http://localhost:8000/health` — it should show `{"status": "healthy"}`
3. Make sure your OpenAI API key has credits/billing enabled

### "Python 3.11+ required"
**Cause**: Your Python version is too old.
**Fix**: Install Python 3.11 or newer (see Prerequisites section above).

### Windows: "running scripts is disabled on this system"
**Cause**: PowerShell execution policy blocks the virtual environment activation.
**Fix**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating the virtual environment again.

---

## Running with Docker (Alternative)

If you have Docker installed, you can skip all the Python setup:

```bash
# Build and start (one command)
docker compose up --build -d

# Check if it's running
docker compose ps

# View logs
docker compose logs -f app

# Stop
docker compose down
```

Then open `http://localhost:8000` in your browser.

---

## Stopping the Server

To stop the server, go to the terminal where it's running and press:
- **Mac/Linux**: `Ctrl+C`
- **Windows**: `Ctrl+C`

To deactivate the virtual environment:
```bash
deactivate
```

---

## Quick Start Summary (TL;DR)

```bash
# 1. Unzip and open folder in VS Code
# 2. Open terminal in VS Code (Ctrl+`)
# 3. Set up:
python3 -m venv .venv
source .venv/bin/activate          # Mac/Linux
pip install -r requirements.txt

# 4. Edit .env → put your OPENAI_API_KEY

# 5. Run:
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 6. Open http://localhost:8000 in browser
# 7. Upload files and start chatting!
```

---

*If you have any issues, check the Common Issues section above or look at the terminal output for error messages.*
