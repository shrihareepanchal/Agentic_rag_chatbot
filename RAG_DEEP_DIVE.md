# Understanding RAG — Retrieval-Augmented Generation

## A Deep Dive into How RAG Works in This Project (With Code)

---

## Table of Contents

1. What is RAG?
2. Why RAG Exists — The Problem It Solves
3. The Three Pillars of RAG
4. Pillar 1 — Retrieval 
   - 4.1 Document Ingestion — Preparing Data for Retrieval
   - 4.2 Chunking — Breaking Documents into Searchable Pieces
   - 4.3 Embedding — Converting Text to Vectors
   - 4.4 Storing — Saving Vectors in ChromaDB
   - 4.5 Searching — Finding Relevant Chunks at Query Time
   - 4.6 Re-Ranking — Refining Results with RRF
5. [Pillar 2 — Augmentation 
   - [5.1 System Prompt Construction
   - [5.2 Dynamic Context Injection
   - [5.3 Tool Results as Augmented Context
   - [5.4 Token-Aware History Trimming](#54-token-aware-history-trimming)
6. [Pillar 3 — Generation (Producing the Final Answer)](#6-pillar-3--generation-producing-the-final-answer)
   - [6.1 The LLM Reasoning Step](#61-the-llm-reasoning-step)
   - [6.2 Grounded Answer Generation](#62-grounded-answer-generation)
   - [6.3 Follow-Up Suggestion Generation](#63-follow-up-suggestion-generation)
7. [The Agentic Loop — How R, A, and G Work Together](#7-the-agentic-loop--how-r-a-and-g-work-together)
8. [Standard RAG vs. Agentic RAG](#8-standard-rag-vs-agentic-rag)
9. [End-to-End Walkthrough — A Real Query](#9-end-to-end-walkthrough--a-real-query)
10. [Key Concepts Glossary](#10-key-concepts-glossary)
11. [Document Updates & Re-Ingestion — Handling Changing Documents](#11-document-updates--re-ingestion--handling-changing-documents)
    - [11.1 The Problem — Documents Change Over Time](#111-the-problem--documents-change-over-time)
    - [11.2 How This RAG Solves It — Replace-on-Reupload Strategy](#112-how-this-rag-solves-it--replace-on-reupload-strategy)
    - [11.3 The Re-Ingestion Pipeline (Step by Step)](#113-the-re-ingestion-pipeline-step-by-step)
    - [11.4 SQL Table Updates for Structured Data](#114-sql-table-updates-for-structured-data)
    - [11.5 Manual Deletion via API](#115-manual-deletion-via-api)
    - [11.6 Source Cache Invalidation](#116-source-cache-invalidation)
12. [Multi-File Upload — Processing Multiple Documents at Once](#12-multi-file-upload--processing-multiple-documents-at-once)
    - [12.1 The Multi-File Upload Endpoint](#121-the-multi-file-upload-endpoint)
    - [12.2 How It Works — End-to-End Flow](#122-how-it-works--end-to-end-flow)
    - [12.3 Frontend Multi-File Support](#123-frontend-multi-file-support)
    - [12.4 Cross-File Retrieval — Querying Across All Documents](#124-cross-file-retrieval--querying-across-all-documents)

---

## 1. What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It is a technique that improves the accuracy and reliability of Large Language Model (LLM) responses by grounding them in external, factual data.

Break it down word by word:

| Word | Meaning | In This Project |
|------|---------|-----------------|
| **Retrieval** | Searching through a knowledge base to find relevant information | Querying ChromaDB for document chunks that match the user's question |
| **Augmented** | Enhancing or enriching the LLM's input with the retrieved information | Injecting retrieved chunks, system prompt, source lists, and SQL schemas into the LLM's context window |
| **Generation** | Using the LLM to produce a natural language answer based on the augmented context | GPT-4o-mini reads the retrieved chunks and generates a grounded, cited response |

**In simple terms:** Instead of asking the LLM to answer from its training data alone (which can be outdated, incomplete, or hallucinated), RAG first *retrieves* relevant documents, *augments* the LLM's prompt with those documents, and then lets the LLM *generate* an answer that's grounded in actual data.

---

## 2. Why RAG Exists — The Problem It Solves

LLMs have three critical limitations that RAG addresses:

### Problem 1: Knowledge Cutoff
LLMs are trained on data up to a certain date. They don't know about your company's internal documents, your latest reports, or yesterday's data.

**RAG solution:** Upload your documents → they become the LLM's knowledge base.

### Problem 2: Hallucination
LLMs can confidently generate false information — invented statistics, fake URLs, wrong dates.

**RAG solution:** The LLM is instructed to ONLY use information from retrieved documents. The system prompt enforces this:

```python
# backend/agents/graph.py — SYSTEM_PROMPT (lines 320-330)
SYSTEM_PROMPT = """...
## CORE RULE — ZERO HALLUCINATION
**Every sentence you write MUST be directly supported by text in the retrieved chunks.**
- If the document says it, you can say it.
- If the document does NOT say it, you MUST NOT say it...
"""
```

### Problem 3: No Source Attribution
When an LLM answers from training data, you can't verify *where* it got the information.

**RAG solution:** Every answer includes citations — the source document, chunk ID, page number, and relevance score:

```python
# backend/models/schemas.py — Citation model
class Citation(BaseModel):
    source: str         # "report.pdf"
    chunk_id: str       # "report.pdf::chunk_3"
    page_number: int | None  # 3
    score: float        # 0.82
    snippet: str        # "The revenue grew by 15%..."
```

---

## 3. The Three Pillars of RAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                                   │
│                                                                         │
│   ┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│   │   RETRIEVAL      │   │   AUGMENTATION    │   │   GENERATION      │   │
│   │                  │   │                   │   │                   │   │
│   │  "Find the       │──▶│  "Feed it to      │──▶│  "Let the LLM     │   │
│   │   right data"    │   │   the LLM with    │   │   write the       │   │
│   │                  │   │   instructions"   │   │   answer"          │   │
│   │  • Load docs     │   │                   │   │                   │   │
│   │  • Chunk text    │   │  • System prompt  │   │  • LLM reasoning  │   │
│   │  • Embed vectors │   │  • Source lists   │   │  • Grounded text  │   │
│   │  • Search DB     │   │  • Retrieved text │   │  • Citations      │   │
│   │  • Re-rank       │   │  • SQL schemas    │   │  • Follow-ups     │   │
│   │                  │   │  • History trim   │   │                   │   │
│   └─────────────────┘   └──────────────────┘   └───────────────────┘   │
│                                                                         │
│         OFFLINE                    ONLINE                ONLINE          │
│     (during upload)           (during query)        (during query)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Retrieval** has an offline phase (ingestion) and an online phase (search). **Augmentation** and **Generation** happen in real time during every query.

---

## 4. Pillar 1 — Retrieval (Finding the Right Information)

Retrieval is the foundation of RAG. If you retrieve the wrong chunks, the LLM will generate a wrong answer. This project's retrieval pipeline has two phases:

- **Offline phase (ingestion):** Prepare documents for fast searching later.
- **Online phase (query-time):** Quickly find the most relevant chunks when a user asks a question.

---

### 4.1 Document Ingestion — Preparing Data for Retrieval

When a user uploads a file, the system processes it through a 5-step pipeline defined in the indexer:

```python
# backend/ingestion/indexer.py — ingest_file() (the main pipeline)
def ingest_file(file_path: str, original_filename: str | None = None) -> int:
    file_name = original_filename or Path(file_path).name
    source_id = _make_source_id(file_name)  # SHA1[:12] hash of filename

    # Step 1: DEDUPLICATION — Remove old chunks if file was previously uploaded
    store = get_vector_store()
    store.delete_by_source(file_name)

    # Step 2: LOAD — Convert file to (text, metadata) tuples
    documents = load_file(file_path)

    # Step 3: CHUNK — Split text into small, searchable pieces
    settings = get_settings()
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    # Step 4: EMBED — Convert each chunk's text into a numerical vector
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    # Step 5: STORE — Save chunks + vectors in ChromaDB
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    # Bonus: Load CSV/JSON into SQLite for SQL queries
    _load_into_sql(file_path, file_name)

    return len(chunks)
```

**Why 5 steps?** Each step prepares the data for efficient semantic search:
- **Dedup** prevents duplicate chunks when re-uploading.
- **Load** handles 6 file formats (PDF, TXT, DOCX, JSON, CSV, MD).
- **Chunk** makes documents searchable at a granular level.
- **Embed** converts human language into math (vectors) that machines can compare.
- **Store** puts everything in a database optimized for similarity search.

---

### 4.2 Chunking — Breaking Documents into Searchable Pieces

**Why chunk?** A 50-page PDF doesn't fit in the LLM's context window. Even if it did, searching a 50-page document for one sentence would be slow and imprecise. Chunking breaks it into small, focused pieces that can be independently retrieved.

```python
# backend/ingestion/chunker.py — chunk_documents()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,        # 512 characters (configurable)
    chunk_overlap=chunk_overlap,  # 64 characters overlap
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**How `RecursiveCharacterTextSplitter` works:**
1. First tries to split on `\n\n` (paragraph boundaries) — preserves full paragraphs.
2. If a paragraph is too long (>512 chars), tries `\n` (line breaks).
3. If a line is too long, tries `. ` (sentence boundaries).
4. If a sentence is too long, tries ` ` (word boundaries).
5. Last resort: splits on individual characters.

**This hierarchy ensures that chunks are semantically meaningful** — paragraph-level when possible, sentence-level only when necessary.

**Overlap (64 chars):** Consecutive chunks share 64 characters of text. This prevents information loss at chunk boundaries. If a key fact spans two chunks, the overlap ensures it appears fully in at least one:

```
Chunk 1: "The company reported Q3 revenue of $2.5 billion, exceeding ||analyst expectations..."
Chunk 2:                                                   "...exceeding ||analyst expectations by 12%, driven by strong cloud growth."
                                                              ↑ overlap zone ↑
```

**Contextual prefix:** Each chunk is labeled with its source document:

```python
# backend/ingestion/chunker.py — _build_context_prefix()
def _build_context_prefix(metadata: Dict) -> str:
    parts = []
    if metadata.get("source"):
        parts.append(f"Document: {metadata['source']}")
    if metadata.get("doc_type"):
        parts.append(f"Type: {metadata['doc_type']}")
    if metadata.get("page_number"):
        parts.append(f"Page: {metadata['page_number']}")
    return f"[{' | '.join(parts)}] " if parts else ""

# Result: "[Document: report.pdf | Type: pdf | Page: 3] The company reported..."
```

**Why prefix?** When the embedding model converts this text to a vector, the prefix information becomes part of the mathematical representation. This means searching for "report.pdf page 3 revenue" will naturally match this chunk.

---

### 4.3 Embedding — Converting Text to Vectors

**This is the mathematical core of RAG retrieval.**

An **embedding** is a list of numbers (a vector) that captures the *meaning* of text. Texts with similar meanings produce vectors that are close together in mathematical space.

```
"The cat sat on the mat"  →  [0.12, -0.45, 0.78, 0.33, ...]  (1536 numbers)
"A feline rested on a rug" →  [0.11, -0.44, 0.79, 0.31, ...]  (very similar!)
"Stock market crashed"    →  [0.89, 0.22, -0.56, 0.01, ...]  (very different)
```

#### 2D Sketch — How a Query Finds Relevant Chunks

In reality, embeddings live in 1536-dimensional space. But if we project them down to **2D**, you can visualize how a query vector lands near semantically similar chunks and far from unrelated ones:

```
                        ▲ Dimension 2 (meaning axis)
                        │
                   1.0  │          ⭐ QUERY: "What was Q3 revenue?"
                        │            ╲
                        │             ╲  (small distance = HIGH similarity)
                   0.8  │              ╲
                        │            ● chunk_2: "Q3 revenue was $2.5B..."
                        │                  (distance=0.18, similarity=0.91)
                   0.6  │
                        │        ● chunk_4: "Revenue beat estimates..."
                        │              (distance=0.31, similarity=0.85)
                   0.4  │
                        │
                   0.2  │                              ○ chunk_3: "Employee count grew..."
                        │                                    (distance=1.22, similarity=0.39)
                   0.0  │
                        │  ○ chunk_1: "Board of directors..."
                  -0.2  │       (distance=1.38, similarity=0.31)
                        │
                        │         ○ chunk_0: "Company overview..."
                  -0.4  │              (distance=1.45, similarity=0.28)
                        │
                        └──────────────────────────────────────────► Dimension 1
                       -0.4  -0.2   0.0   0.2   0.4   0.6   0.8  1.0

                  Legend:  ⭐ = Query vector    ● = Relevant chunks (close)    ○ = Irrelevant chunks (far)
```

**How to read this:** The query "What was Q3 revenue?" becomes a point in vector space (⭐). Chunks about revenue (●) cluster near it because their meanings are similar. Chunks about unrelated topics (○) are far away. ChromaDB returns the nearest points.

#### 3D Sketch — Multi-Dimensional Similarity Across Documents

With a third dimension, you can see how chunks from **different uploaded files** form clusters, and how a query can pull relevant chunks from multiple files at once:

```
                            Dimension 3 (topic depth)
                           ╱
                          ╱
                    1.0  ╱
                        ╱  ┌─────────────────────────────────────────────┐
                       ╱   │          3D VECTOR SPACE                    │
                      ╱    │                                             │
                     ╱     │    📄 report.pdf chunks                    │
                    ╱      │    ┌───────────┐                           │
                   ╱       │    │ ●₁ ●₂  ●₃ │  ← revenue, financials   │
                  ╱        │    │   ●₄  ●₅  │                          │
                 ╱   ▲     │    └─────╱─────┘                           │
                ╱    │     │         ╱                                   │
               ╱     │     │    ⭐ QUERY lands HERE                     │
              ╱      │     │     (searches ALL files at once)           │
             ╱       │     │         ╲                                   │
            ╱  Dim 2 │     │    ┌─────╲─────┐                           │
           ╱         │     │    │ ●₆   ●₇  │  ← sales data             │
          ╱          │     │    │  ●₈  ●₉  │                           │
         ╱           │     │    └───────────┘                           │
        ╱            │     │    📊 financials.csv chunks                │
       ╱             │     │                                             │
      ╱              │     │                         ┌───────────┐      │
     ╱               │     │                         │ ○₁₀  ○₁₁ │      │
    ╱                │     │                         │   ○₁₂    │      │
   ╱                 │     │                         └───────────┘      │
  ╱                  │     │                         📋 handbook.pdf    │
 ╱                   │     │                         (HR topics — far)  │
╱                    │     └─────────────────────────────────────────────┘
─────────────────────┘──────────────────────────────► Dimension 1
```

**How to read this 3D view:**
- Each **uploaded file's** chunks naturally cluster together (same document = similar topics)
- The ⭐ **query** vector lands in the region between `report.pdf` and `financials.csv` because both contain revenue data
- ChromaDB returns the **nearest chunks regardless of which file they came from** (●₁, ●₂, ●₆, ●₇, etc.)
- `handbook.pdf` chunks (○) are far away in vector space because HR topics are semantically different from revenue
- This is how **cross-file retrieval** works: one query, one vector space, chunks from all files compete on relevance

**The real math:** In actual 1536-dimensional space, "distance" is measured by **cosine similarity** — the angle between two vectors. Two vectors pointing in the same direction (angle ≈ 0°) have similarity ≈ 1.0. Perpendicular vectors (angle = 90°) have similarity ≈ 0.0.

In this project, OpenAI's `text-embedding-3-small` model generates 1536-dimensional vectors:

```python
# backend/ingestion/embedder.py — RealEmbedder
class RealEmbedder:
    _BATCH_SIZE = 100  # Process 100 texts per API call

    def __init__(self, model="text-embedding-3-small", dimensions=1536):
        self._client = OpenAIEmbeddings(model=model, dimensions=dimensions)

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i:i + self._BATCH_SIZE]
            batch_embeddings = self._client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
```

**What happens inside `embed_documents()`:**
1. Each text chunk is sent to OpenAI's embedding API.
2. The API's neural network converts the text into a 1536-dimensional vector.
3. Each dimension captures some aspect of meaning (there's no single human-interpretable meaning per dimension — they work together).

**Batching:** Instead of one API call per chunk (slow, rate-limited), texts are processed in batches of 100 for efficiency.

**Mock fallback:** For development without an API key, a mock embedder creates deterministic pseudo-embeddings using MD5 hashing:

```python
# backend/ingestion/embedder.py — MockEmbedder
def embed(self, texts):
    for text in texts:
        hash_bytes = hashlib.md5(text.encode()).digest()
        rng = random.Random(int.from_bytes(hash_bytes[:4], "big"))
        raw = [rng.gauss(0, 1) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(x * x for x in raw))
        normalized = [x / norm for x in raw]  # Unit-norm vector
```

---

### 4.4 Storing — Saving Vectors in ChromaDB

ChromaDB is a **vector database** — a database optimized for storing and searching high-dimensional vectors. Think of it as a "semantic search engine" for your documents.

```python
# backend/retrieval/vector_store.py — ChromaVectorStoreAdapter.add()
def add(self, ids, texts, embeddings=None, metadatas=None):
    collection = self._store._collection
    if embeddings is not None:
        # Direct add with pre-computed embeddings
        BATCH = 5000
        for i in range(0, len(ids), BATCH):
            collection.add(
                ids=ids[i:i+BATCH],           # "report.pdf::chunk_0", "report.pdf::chunk_1", ...
                documents=texts[i:i+BATCH],    # The actual text content
                embeddings=embeddings[i:i+BATCH],  # [0.12, -0.45, 0.78, ...] for each chunk
                metadatas=metadatas[i:i+BATCH],     # {"source": "report.pdf", "page_number": 1}
            )
```

**What gets stored in ChromaDB for each chunk:**

| Field | Example | Purpose |
|-------|---------|---------|
| `id` | `report.pdf::chunk_3` | Unique identifier |
| `document` | `[Document: report.pdf] The revenue grew...` | Original text (for returning to user) |
| `embedding` | `[0.12, -0.45, 0.78, ...]` (1536 floats) | Vector for similarity search |
| `metadata` | `{"source": "report.pdf", "page_number": 3}` | Filters and attribution |

ChromaDB stores this data persistently on disk at `data/chroma_db/` so it survives server restarts.

---

### 4.5 Searching — Finding Relevant Chunks at Query Time

When a user asks a question, the **same embedding model** converts their question into a vector, then ChromaDB finds the stored chunks with the *most similar* vectors:

```python
# backend/agents/tools.py — retrieve_documents tool (lines 166-180)
@tool
def retrieve_documents(query: str, top_k: int = 10, source_filter: str = "") -> str:
    """Search the knowledge base for text chunks semantically relevant to the query."""
    
    vs = get_vector_store()
    
    # ChromaDB embeds the query using the same model, then finds nearest vectors
    docs_and_scores = vs.similarity_search_with_score(query_str, k=actual_k, filter=chroma_filter)
```

**How similarity search works under the hood:**

```
User query: "What was the Q3 revenue?"
    ↓
Embed query → [0.15, -0.42, 0.80, 0.30, ...]   (1536-dim vector)
    ↓
Compare with ALL stored vectors using cosine similarity:

  chunk_0: "Company overview..."        → distance = 1.45  (far away = irrelevant)
  chunk_1: "Board of directors..."       → distance = 1.38  (far away)
  chunk_2: "Q3 revenue was $2.5B..."     → distance = 0.18  (very close = relevant!)
  chunk_3: "Employee count grew..."      → distance = 1.22  (far away)
  chunk_4: "Revenue beat estimates..."   → distance = 0.31  (close = relevant)
    ↓
Return top-k (10) most similar chunks, sorted by distance
```

**Distance to similarity conversion:**

```python
# backend/agents/tools.py — inside retrieve_documents (lines 200-214)
# ChromaDB returns cosine distance (0 = identical, 2 = opposite)
# Convert to similarity score (1 = identical, 0 = opposite)
dist = float(raw_score)
if dist <= 2.0:
    score = round(max(0.0, 1.0 - dist / 2.0), 4)  # 0.18 distance → 0.91 similarity
else:
    score = round(max(0.0, 1.0 / (1.0 + dist)), 4)
```

**Optional source filtering:**

```python
# Filter results to a specific document
chroma_filter = None
if source_filter:
    chroma_filter = {"source": sf}  # Only search within "report.pdf"
```

This lets the agent restrict retrieval to a specific uploaded file when the user asks about "the file I uploaded."

---

### 4.6 Re-Ranking — Refining Results with RRF

Raw semantic search isn't perfect. Two texts can be semantically similar but about different topics ("Python the snake" vs. "Python the language"). This project uses **Reciprocal Rank Fusion (RRF)** to combine two ranking signals:

1. **Semantic rank** — How mathematically similar is the chunk vector to the query vector?
2. **Keyword rank** — How many of the query's actual words appear in the chunk?

```python
# backend/agents/tools.py — _reciprocal_rank_fusion() (lines 40-85)
def _reciprocal_rank_fusion(semantic_results, query, k_constant=60):
    """
    RRF formula: score = 1/(k + rank_semantic) + 1/(k + rank_keyword)
    """
    # Rank by keyword overlap
    keyword_scores = [
        _keyword_overlap_score(query, r.get("snippet", ""))
        for r in semantic_results
    ]
    keyword_ranked = sorted(range(len(semantic_results)),
                            key=lambda i: keyword_scores[i], reverse=True)

    # Combine ranks using RRF
    rrf_scores = [0.0] * len(semantic_results)
    for rank, idx in enumerate(semantic_ranked):
        rrf_scores[idx] += 1.0 / (k_constant + rank + 1)   # Semantic contribution
    for rank, idx in enumerate(keyword_ranked):
        rrf_scores[idx] += 1.0 / (k_constant + rank + 1)   # Keyword contribution

    # Sort by combined RRF score
    reranked = sorted(range(len(semantic_results)),
                      key=lambda i: rrf_scores[i], reverse=True)
```

**Example of RRF in action:**

```
Query: "What is the Python release schedule?"

Semantic search returns (sorted by vector similarity):
  Rank 1: "Python 3.12 was released in October 2023..."     (correct!)
  Rank 2: "The python snake is found in tropical regions..." (wrong topic!)
  Rank 3: "Python development follows PEP cycles..."        (correct!)

Keyword overlap ranking (word matching):
  Rank 1: "Python development follows PEP cycles. The release schedule..." (has "release schedule"!)
  Rank 2: "Python 3.12 was released in October 2023..."     (has "Python", "release")
  Rank 3: "The python snake is found in tropical regions..." (only has "Python")

RRF combined (k=60):
  "Python development follows PEP cycles..." → 1/(60+1) + 1/(60+1) = 0.0328  (TOP)
  "Python 3.12 was released..."              → 1/(60+1) + 1/(60+2) = 0.0325
  "The python snake..."                       → 1/(60+2) + 1/(60+3) = 0.0321  (BOTTOM)
```

The snake result gets pushed down because it lacks the keyword "release schedule," even though it was semantically similar (both mention "Python").

**Additionally**, the system detects when retrieved chunks may be about the wrong topic entirely:

```python
# backend/agents/tools.py — retrieve_documents (lines 240-265)
# If 60%+ of query keywords are missing from ALL retrieved chunks,
# warn the LLM that results may be irrelevant
missing_ratio = len(missing_terms) / len(key_terms)
if missing_ratio >= 0.6:
    query_term_warning = (
        f"STOP — TOPIC MISMATCH DETECTED: The key query terms {missing_terms} "
        f"do NOT appear in ANY retrieved chunk..."
    )
```

This is a safety net that prevents the "Augmentation" and "Generation" stages from working with irrelevant data.

---

## 5. Pillar 2 — Augmentation (Enhancing the LLM's Context)

**Augmentation is the bridge between Retrieval and Generation.** It's the process of constructing the prompt that the LLM will use to generate its answer. This is where all retrieved information, instructions, and context are assembled into a single, structured input.

Without augmentation, the LLM would just get the user's question and answer from training data — which is just a regular chatbot, not RAG.

---

### 5.1 System Prompt Construction

The system prompt is the foundation of augmentation. It tells the LLM *how* to behave, *what rules* to follow, and *what tools* it has access to:

```python
# backend/agents/graph.py — SYSTEM_PROMPT (lines 318-460)
SYSTEM_PROMPT = """You are a friendly, accurate AI knowledge assistant.
You answer questions **strictly from the content of retrieved documents** — nothing more, nothing less.

## CORE RULE — ZERO HALLUCINATION
**Every sentence you write MUST be directly supported by text in the retrieved chunks.**

## GROUNDING RULES
1. **ONLY** use information found in the retrieved document chunks. NEVER use your training knowledge.
2. If nothing relevant is found, say: "I couldn't find that in the uploaded documents."
3. **NEVER fabricate** facts, numbers, dates, names, URLs, links, or explanations.
...
"""
```

**Why this matters for augmentation:** The system prompt constrains the Generation step. No matter what the LLM "knows" from training, it's forced to only use retrieved content. This is the mechanism that makes RAG accurate.

---

### 5.2 Dynamic Context Injection

The `agent_node` augments the base system prompt with runtime information — what documents exist, what SQL tables are available, and what the user has recently uploaded:

```python
# backend/agents/graph.py — agent_node() (lines 565-620)
def agent_node(state: AgentState) -> Dict:
    dynamic_prompt = SYSTEM_PROMPT  # Start with base prompt

    # AUGMENTATION 1: Inject indexed source filenames
    source_names = _get_cached_source_names()
    if source_names:
        sources_block = "\n## INDEXED DOCUMENTS (available for retrieval)\n"
        for s in source_names:
            sources_block += f"- `{s}`\n"
        dynamic_prompt += sources_block

    # AUGMENTATION 2: Inject recently uploaded files with priority markers
    active = state.get("active_sources", [])
    if active:
        sources_block += f"\n**MOST RECENT UPLOAD**: `{active[-1]}`\n"
        dynamic_prompt += sources_block

    # AUGMENTATION 3: Inject SQL table schemas
    sql_store = get_sql_store()
    schema = sql_store.get_schema()
    for table_name, cols in schema.items():
        col_names = ", ".join(c["name"] for c in cols[:15])
        sql_block += f"- **{table_name}**: columns = [{col_names}]\n"
    dynamic_prompt += sql_block
```

**What the augmented prompt looks like to the LLM:**

```
┌────────────────────────────────────────────────┐
│ SYSTEM PROMPT (~4000 tokens)                   │
│  ├── Core rules (zero hallucination)           │
│  ├── Topic matching rules                      │
│  ├── Tool usage instructions                   │
│  ├── Formatting guidelines                     │
│  │                                             │
│  ├── ## INDEXED DOCUMENTS               ← NEW │
│  │   - report.pdf                              │
│  │   - sales_data.csv                          │
│  │   - meeting_notes.docx                      │
│  │                                             │
│  ├── ## RECENTLY UPLOADED                ← NEW │
│  │   - sales_data.csv ← MOST RECENT UPLOAD    │
│  │                                             │
│  └── ## SQL TABLES                       ← NEW │
│      - sales_data: columns = [date, product,   │
│                                revenue, qty]    │
├────────────────────────────────────────────────┤
│ CONVERSATION HISTORY (trimmed to fit)          │
│  ├── Human: "What was Q3 revenue?"             │
│  ├── AI: [tool_call: retrieve_documents]       │
│  ├── Tool: {results: [...chunks...]}           │
│  └── Human: "Break it down by product"         │
├────────────────────────────────────────────────┤
│ NEW USER MESSAGE                               │
│  └── "What about Q4 forecast?"                 │
└────────────────────────────────────────────────┘
```

This entire structure IS the augmentation — the LLM receives not just the user's question, but a rich context of available data, rules, and retrieved information.

---

### 5.3 Tool Results as Augmented Context

The most important augmentation happens when tool results (retrieved chunks) are injected into the conversation. This is the core of RAG — after retrieval, the tool results become part of the LLM's context:

```python
# The flow:
# 1. LLM decides to call retrieve_documents
# 2. Tool executes the search and returns JSON
# 3. The JSON result becomes a ToolMessage in the conversation
# 4. The LLM reads the ToolMessage and generates an answer grounded in it

# What the tool returns (backend/agents/tools.py):
{
    "results": [
        {
            "source": "report.pdf",
            "chunk_id": "report.pdf::chunk_12",
            "page_number": 5,
            "score": 0.87,
            "snippet": "[Document: report.pdf | Type: pdf | Page: 5] Q3 revenue reached $2.5 billion, 
                        a 15% increase year-over-year driven by cloud services growth..."
        },
        {
            "source": "report.pdf",
            "chunk_id": "report.pdf::chunk_13",
            "page_number": 5,
            "score": 0.72,
            "snippet": "[Document: report.pdf | Type: pdf | Page: 5] The cloud division alone contributed 
                        $1.2 billion, up from $950 million in Q2..."
        }
    ],
    "count": 2,
    "query_term_match": true
}
```

**This JSON becomes the "Retrieved" part of "Retrieval-Augmented Generation."** The LLM literally reads these snippets and uses them as its knowledge base to generate the answer.

---

### 5.4 Token-Aware History Trimming

The context window (how much text the LLM can read at once) is limited. Augmentation must fit everything within this window:

```python
# backend/agents/graph.py — _trim_messages_to_token_limit() (lines 118-180)
def _trim_messages_to_token_limit(messages, max_tokens=60000):
    """
    Trim conversation history to fit within token budget.
    Always keeps the LAST message (current user query).
    Removes oldest messages first while preserving most recent context.
    
    IMPORTANT: Preserves tool_call / tool_response pairs — an AIMessage with
    tool_calls and its subsequent ToolMessage(s) are never separated.
    """
```

**Token budget breakdown:**
```
Total budget: 60,000 tokens
  - System prompt: ~4,000 tokens (reserved)
  - LLM response:  ~4,000 tokens (reserved)
  - History:       ~52,000 tokens (available for conversation + tool results)
```

If the conversation exceeds 52,000 tokens, the oldest messages are removed first. This ensures the most *recent* retrievals and context are always available to the LLM.

---

## 6. Pillar 3 — Generation (Producing the Final Answer)

Generation is where the LLM reads all the augmented context and produces a human-readable answer. In a RAG system, generation is *constrained* — the LLM must only use information from the retrieved chunks.

---

### 6.1 The LLM Reasoning Step

The LLM (GPT-4o-mini) receives the full augmented context and makes a decision:

```python
# backend/agents/graph.py — agent_node() (lines 635-650)
# Build the full message list
messages_with_system = [SystemMessage(content=dynamic_prompt)] + trimmed_messages

# The LLM can either:
#   A) Call a tool (retrieve_documents, query_database, etc.) to get more info
#   B) Generate a final text answer from the context it already has
response = llm_with_tools.invoke(messages_with_system)
```

The LLM has tools bound to it, so each response is either:
- **Tool calls** (the LLM needs more information → loop back to Retrieval)
- **Text content** (the LLM has enough information → proceed to output)

This is what makes this system **Agentic** RAG — the LLM can call tools multiple times, building up context iteratively before generating its final answer.

---

### 6.2 Grounded Answer Generation

When the LLM generates its final answer, several post-processing steps ensure quality:

**Citation extraction** — Identifies which retrieved chunks were used:

```python
# backend/agents/graph.py — extract_citations_from_tools() (lines 656-680)
def extract_citations_from_tools(state: AgentState) -> Dict:
    citations = list(state.get("citations", []))
    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        data = json.loads(msg.content)
        if "results" in data:
            for r in data["results"]:
                if r.get("score", 0) > 0:
                    citations.append(r)  # Track every retrieved chunk as a citation
    return {"citations": citations}
```

**Confidence scoring** — Measures how well the retrieved chunks matched the query:

```python
# backend/agents/graph.py — output_guard_node() (lines 710-730)
# Filter noise: remove citations with scores below 0.10
cits = [c for c in raw_cits if c.get("score", 0) >= 0.10]

# Calculate confidence
scores = [c.get("score", 0) for c in cits]
avg_score = sum(scores) / len(scores)
max_score = max(scores)

# Weighted: 60% best match + 40% average
confidence = round(0.6 * max_score + 0.4 * avg_score, 3)
```

**Low confidence override** — If the retrieval quality is poor, the system overrides the LLM's answer:

```python
# backend/agents/graph.py — output_guard_node() (lines 735-745)
if confidence < settings.confidence_threshold and raw_cits:
    if max(c.get("score", 0) for c in raw_cits) < settings.confidence_threshold:
        final_answer = (
            "I don't have enough information in the indexed knowledge base "
            "to answer this question with confidence..."
        )
```

This is a Generation-phase guardrail: even if the LLM produces a fluent answer, the system replaces it if the underlying retrieval was poor.

---

### 6.3 Follow-Up Suggestion Generation

After the main answer, a **separate LLM call** generates contextual follow-up questions grounded in the retrieved chunks:

```python
# backend/agents/graph.py — run_agent() (lines 930-970)
# Build grounded follow-up prompt
_followup_messages = [
    _SM(content=(
        "You generate exactly 3 short follow-up questions. Rules:\n"
        "- Each question MUST be answerable DIRECTLY from the DOCUMENT CHUNKS below\n"
        "- ONLY ask about topics that appear VERBATIM in the chunks\n"
        "- Pick 3 DIFFERENT topics from the chunks the user hasn't asked about yet\n"
    )),
    _HM(content=(
        f"User asked: {message}\n\n"
        f"Document source: {_cited_source}\n\n"
        f"Retrieved chunks:\n{_chunk_snippets[:3000]}\n\n"
        f"Assistant answered:\n{answer[:500]}"
    )),
]

# Async with 4-second timeout — don't delay the main response
followup_resp = await asyncio.wait_for(
    followup_llm.ainvoke(_followup_messages),
    timeout=4.0,
)
```

**This is also augmentation + generation** — the follow-up LLM receives the retrieved chunks (augmentation) and generates questions from them (generation). The questions are grounded in the actual document content, not invented.

---

## 7. The Agentic Loop — How R, A, and G Work Together

In standard RAG, the pipeline runs once: Retrieve → Augment → Generate. In **Agentic RAG**, the agent can loop through retrieval and augmentation multiple times before generating the final answer.

```python
# backend/agents/graph.py — build_agent_graph() (lines 755-790)
def build_agent_graph():
    builder = StateGraph(AgentState)

    # 5 nodes
    builder.add_node("input_guard", input_guard_node)      # Pre-processing
    builder.add_node("agent", agent_node)                  # AUGMENTATION + decision
    builder.add_node("tools", ToolNode(ALL_TOOLS))         # RETRIEVAL (tool execution)
    builder.add_node("extract_citations", extract_citations_from_tools)
    builder.add_node("output_guard", output_guard_node)    # GENERATION post-processing

    # The agentic loop
    builder.add_edge(START, "input_guard")
    builder.add_edge("input_guard", "agent")

    # KEY: Agent decides — call tools (RETRIEVAL) or generate answer?
    builder.add_conditional_edges(
        "agent",
        tools_condition,  # If LLM response has tool_calls → go to tools
        {
            "tools": "tools",   # More retrieval needed
            END: "output_guard", # Ready to generate final answer
        },
    )

    # After tools run → extract citations → back to agent for more reasoning
    builder.add_edge("tools", "extract_citations")
    builder.add_edge("extract_citations", "agent")  # ← THE LOOP
    builder.add_edge("output_guard", END)
```

**The loop visualized:**

```
User: "Summarize sales_data.csv and compare with report.pdf"

Loop 1 (RETRIEVAL): Agent calls get_database_schema → learns table structure
Loop 2 (RETRIEVAL): Agent calls query_database → gets COUNT, AVG stats from CSV
Loop 3 (RETRIEVAL): Agent calls retrieve_documents → gets report.pdf chunks
Loop 4 (GENERATION): Agent has enough context → writes comparison answer
```

Each loop adds more information to the augmented context. By loop 4, the agent has SQL results AND document chunks, enabling a comprehensive answer.

---

## 8. Standard RAG vs. Agentic RAG

| Aspect | Standard RAG | Agentic RAG (This Project) |
|--------|-------------|---------------------------|
| Retrieval | Single query, single search | Multiple tools, multiple searches |
| Decision logic | Fixed pipeline | LLM decides what to retrieve next |
| Data sources | Usually one (vector DB) | Vector DB + SQL DB + web scraping |
| Re-ranking | Often none | RRF (semantic + keyword fusion) |
| Iteration | One-shot | Loops until enough context gathered |
| Source filtering | Rarely | Dynamic per-query based on user intent |
| Error recovery | None | Agent can retry with different query |
| SQL support | No | Agent writes and executes SQL queries |

**This project is "Agentic" because the LLM:**
1. **Chooses** which tool to call (vector search, SQL query, full-doc retrieval, clarification).
2. **Decides** when it has enough information to answer.
3. **Adapts** its search strategy based on previous results.
4. **Loops** through retrieval multiple times if needed.

---

## 9. End-to-End Walkthrough — A Real Query

Let's trace exactly what happens when a user asks: **"What was the total revenue in Q3?"** after uploading `quarterly_report.pdf`.

### Step 1: HTTP Request
```
POST /chat
{
    "message": "What was the total revenue in Q3?",
    "session_id": "abc-123",
    "active_sources": ["quarterly_report.pdf"]
}
```

### Step 2: Input Guard (Pre-processing)
```python
# input_guard_node checks:
# 1. Is the query in broken English? → No, it's clean → skip rewriting
# 2. Does it contain PII? → No SSN, email, etc. → pass through
# 3. Is it a prompt injection? → No "ignore instructions" pattern → pass through
```

### Step 3: Agent Node — First Pass (AUGMENTATION)
The LLM receives:
- **System prompt** with zero-hallucination rules
- **Injected source list**: `quarterly_report.pdf` (marked as most recent upload)
- **Injected SQL tables**: (none for PDF)
- **User message**: "What was the total revenue in Q3?"

The LLM **decides** to call `retrieve_documents`:
```json
{"tool_calls": [{"name": "retrieve_documents", "args": {
    "query": "total revenue Q3",
    "source_filter": "quarterly_report.pdf"
}}]}
```

### Step 4: Tool Execution (RETRIEVAL)
```python
# retrieve_documents tool:
# 1. Embeds "total revenue Q3" → [0.15, -0.42, 0.80, ...] (1536 dims)
# 2. Searches ChromaDB with source_filter="quarterly_report.pdf"
# 3. Finds top 10 matching chunks
# 4. Re-ranks with RRF (semantic + keyword fusion)
# 5. Returns JSON with chunks, scores, snippets
```

**Result:**
```json
{
    "results": [
        {
            "source": "quarterly_report.pdf",
            "chunk_id": "quarterly_report.pdf::chunk_12",
            "page_number": 5,
            "score": 0.87,
            "snippet": "[Document: quarterly_report.pdf | Page: 5] Total Q3 revenue reached $2.5 billion, representing a 15% YoY increase..."
        }
    ],
    "count": 5,
    "query_term_match": true
}
```

### Step 5: Citation Extraction
```python
# extract_citations_from_tools parses the tool result:
# citations = [{"source": "quarterly_report.pdf", "score": 0.87, "snippet": "..."}]
```

### Step 6: Agent Node — Second Pass (AUGMENTATION + GENERATION)
Now the LLM has:
- System prompt + source list (same as before)
- **NEW: ToolMessage with retrieved chunks** ← This is the key augmentation
- User's original question

The LLM reads the chunks and **generates**:

```
"Based on the quarterly report, **total Q3 revenue reached $2.5 billion**,
representing a **15% year-over-year increase**. This growth was primarily
driven by the cloud services division."
```

### Step 7: Output Guard (Post-processing)
```python
# output_guard_node:
# 1. PII check on output → clean
# 2. Filter citations below 0.10 score → all pass
# 3. Confidence = 0.6 × 0.87 + 0.4 × 0.65 = 0.782 → good confidence
# 4. No low-confidence override needed
```

### Step 8: Follow-Up Generation
```python
# Separate LLM call generates 3 follow-up questions from the chunks:
# "What drove the cloud services growth?"
# "How did Q3 compare to Q2 performance?"
# "What are the projections for Q4?"
```

### Step 9: Final Response
```json
{
    "answer": "Based on the quarterly report, **total Q3 revenue reached $2.5 billion**...",
    "citations": [{"source": "quarterly_report.pdf", "page_number": 5, "score": 0.87}],
    "follow_up_suggestions": [
        "What drove the cloud services growth?",
        "How did Q3 compare to Q2 performance?",
        "What are the projections for Q4?"
    ],
    "confidence": 0.782,
    "pii_detected": false,
    "injection_detected": false
}
```

**Every piece of the answer came from the retrieved chunks, not from the LLM's training data.** That's RAG.

---

## 10. Key Concepts Glossary

| Term | Definition | Where in Code |
|------|-----------|---------------|
| **Embedding** | A list of numbers (vector) that captures the meaning of text. Similar meanings → similar vectors. | `backend/ingestion/embedder.py` |
| **Vector Database** | A database optimized for storing and searching high-dimensional vectors (ChromaDB in this project). | `backend/retrieval/vector_store.py` |
| **Chunk** | A small piece of a document (512 chars max) that can be independently embedded and retrieved. | `backend/ingestion/chunker.py` |
| **Cosine Similarity** | Mathematical measure of angle between two vectors. 1 = identical direction, 0 = perpendicular, -1 = opposite. | Used internally by ChromaDB |
| **Semantic Search** | Finding text by *meaning* rather than exact keyword match. "car" finds "automobile". | `retrieve_documents` tool |
| **RRF (Reciprocal Rank Fusion)** | A technique combining multiple ranking signals (semantic + keyword) into one final ranking. | `_reciprocal_rank_fusion()` in `tools.py` |
| **Context Window** | The maximum amount of text an LLM can process in one call (~128K tokens for GPT-4o-mini). | Managed by `_trim_messages_to_token_limit()` |
| **Grounding** | Constraining the LLM to only use retrieved information, not its training data. | Enforced by `SYSTEM_PROMPT` |
| **Hallucination** | When an LLM generates false information that sounds plausible but isn't supported by data. | Prevented by grounding rules |
| **Tool Calling** | LLM's ability to invoke external functions (search, SQL query) during reasoning. | Agent node in `graph.py` |
| **State Machine** | A graph of nodes where each node transforms the state and edges control flow. | LangGraph's `StateGraph` in `graph.py` |
| **Token** | A sub-word unit (~4 characters in English). LLMs process text as tokens, not characters. | Counted by `tiktoken` in `graph.py` |
| **System Prompt** | Instructions prepended to every LLM call that define behavior, rules, and capabilities. | `SYSTEM_PROMPT` in `graph.py` |
| **Citation** | A reference back to the source document, chunk, page, and relevance score for each piece of information. | `Citation` model in `schemas.py` |
| **Confidence Score** | A 0-1 score measuring how well retrieved chunks matched the query (0.6×max + 0.4×avg). | `output_guard_node()` in `graph.py` |
| **PII** | Personally Identifiable Information (emails, SSNs, phone numbers). Detected and redacted by guardrails. | `backend/guardrails/pii_redactor.py` |
| **Prompt Injection** | Attempts to hijack the LLM by embedding instructions in user input (e.g., "ignore previous instructions"). | `backend/guardrails/injection_detector.py` |
| **SSRF** | Server-Side Request Forgery — exploiting the server to access internal network resources. Blocked by IP range checks. | `_is_url_safe()` in `main.py` |

---

## Summary

RAG is not one technique — it's a **pipeline** of three stages working together:

1. **Retrieval** answers: *"What information is relevant to this question?"*
   - Implemented via: ChromaDB vector search, RRF re-ranking, source filtering, keyword matching

2. **Augmentation** answers: *"How do I present this information to the LLM?"*
   - Implemented via: Dynamic system prompt, source/schema injection, tool results as context, token-aware trimming

3. **Generation** answers: *"What should the final response say?"*
   - Implemented via: GPT-4o-mini with grounding rules, confidence scoring, citation extraction, PII redaction

The **Agentic** aspect adds a fourth dimension — **decision-making** — where the LLM itself chooses *when* to retrieve, *what* to retrieve, and *when* it has enough context to generate a final answer, iterating through the RAG cycle as many times as needed.

---

## 11. Document Updates & Re-Ingestion — Handling Changing Documents

### 11.1 The Problem — Documents Change Over Time

In real-world enterprise use, policy documents, compliance manuals, HR handbooks, and company procedures are updated regularly — quarterly, annually, or even ad-hoc. A RAG system must handle this gracefully:

- **Stale data**: If old chunks remain in the vector database after a document is updated, the LLM may cite outdated policies.
- **Duplicate data**: If new chunks are added without removing old ones, the same document appears twice with conflicting information.
- **Consistency**: Structured data (CSV/JSON in SQL) and unstructured data (vectors in ChromaDB) must both be refreshed.

### 11.2 How This RAG Solves It — Replace-on-Reupload Strategy

This project uses a **filename-based replace-on-reupload** strategy. When you upload a file with the same name as an existing file, the system:

1. **Deletes ALL old chunks** for that filename from ChromaDB
2. **Re-processes** the new file through the full ingestion pipeline (load → chunk → embed → store)
3. **Replaces the SQL table** entirely for structured data (CSV/JSON)

This means: **to update a document, simply re-upload it with the same filename.** The old version is completely replaced.

### 11.3 The Re-Ingestion Pipeline (Step by Step)

Here is the exact code flow when a previously-uploaded document is re-uploaded:

```
User uploads "company_policy_v2.pdf" (same name as existing "company_policy_v2.pdf")
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 0: De-duplicate (DELETE old data)                 │
│                                                         │
│  indexer.py — ingest_file():                           │
│    vs.delete_by_source("company_policy_v2.pdf")        │
│                                                         │
│  This queries ChromaDB:                                │
│    WHERE source = "company_policy_v2.pdf"              │
│    → Gets all matching chunk IDs                       │
│    → Batch-deletes them (5000 per batch)               │
│                                                         │
│  Result: ALL old chunks for this file are gone         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1-4: Normal ingestion pipeline                   │
│                                                         │
│  1. Load: Parse PDF → extract text per page            │
│  2. Chunk: Split into 512-char chunks with overlap     │
│  3. Embed: Generate 1536-dim vectors via OpenAI        │
│  4. Store: Insert new chunks into ChromaDB             │
│                                                         │
│  Each chunk carries metadata:                          │
│    source = "company_policy_v2.pdf"                    │
│    file_type = "pdf"                                   │
│    page_number = 1, 2, 3...                            │
│    chunk_id = "company_policy_v2.pdf::chunk_0", etc.   │
└─────────────────────────────────────────────────────────┘
```

The relevant code in `backend/ingestion/indexer.py`:

```python
# backend/ingestion/indexer.py — ingest_file() Step 0
def ingest_file(file_path: str) -> dict:
    filename = Path(file_path).name
    vs = get_vector_store()

    # ── Step 0: Remove old chunks for this source (de-duplication) ──
    try:
        deleted = vs.delete_by_source(filename)
        if deleted:
            log.info("dedup_deleted", source=filename, chunks_removed=deleted)
    except Exception as e:
        log.warning("dedup_failed", source=filename, error=str(e))

    # Steps 1-4: Load → Chunk → Embed → Store (fresh data)
    ...
```

And the deletion mechanism in `backend/retrieval/vector_store.py`:

```python
# backend/retrieval/vector_store.py — delete_by_source()
def delete_by_source(self, source_filename: str) -> int:
    """Delete all chunks where metadata 'source' matches the filename."""
    results = self._collection.get(where={"source": source_filename})
    ids = results.get("ids", [])
    if ids:
        # Batch delete in groups of 5000
        for i in range(0, len(ids), 5000):
            self._collection.delete(ids=ids[i:i+5000])
    return len(ids)
```

### 11.4 SQL Table Updates for Structured Data

For CSV and JSON files, the update is even simpler — the SQLite table is **completely replaced**:

```python
# backend/ingestion/indexer.py — _load_into_sql()
df.to_sql(table_name, engine, if_exists="replace", index=False)
```

`if_exists="replace"` drops the old table and creates a new one with the updated data. This means:
- Old rows are gone
- New schema is applied (if columns changed)
- The agent's SQL queries always hit the latest data

### 11.5 Manual Deletion via API

You can also manually remove a document from the knowledge base without re-uploading:

```
DELETE /sources/{filename}
```

This:
1. Removes all vector chunks from ChromaDB where `source = filename`
2. Deletes the uploaded file from disk
3. Returns the count of deleted chunks

### 11.6 Source Cache Invalidation

After any upload, re-upload, or deletion, the system calls `invalidate_source_cache()` to clear the 30-second TTL cache of source names. This ensures the agent's system prompt immediately reflects the updated list of available documents:

```python
# backend/agents/graph.py
def invalidate_source_cache():
    _get_source_names.cache_clear()
```

**Summary of the update strategy:**

| Scenario | What Happens |
|----------|-------------|
| Upload new file `report.pdf` | Ingested as new — chunks added to ChromaDB |
| Re-upload same `report.pdf` | Old chunks deleted → new chunks ingested (full replace) |
| Delete via `DELETE /sources/report.pdf` | All chunks removed from ChromaDB + file deleted from disk |
| Re-upload CSV `data.csv` | Old vector chunks deleted + SQLite table replaced with new data |
| Source cache | Invalidated immediately so agent sees changes |

---

## 12. Multi-File Upload — Processing Multiple Documents at Once

### 12.1 The Multi-File Upload Endpoint

The system supports uploading multiple files in a single request via the `POST /upload/batch` endpoint:

```
POST /upload/batch
Content-Type: multipart/form-data

files: [file1.pdf, file2.csv, file3.txt, ...]
```

**Response** (`MultiUploadResponse`):

```json
{
  "status": "success",
  "total_files": 3,
  "successful": 2,
  "failed": 1,
  "results": [
    {
      "status": "success",
      "filename": "file1.pdf",
      "source_id": "a1b2c3d4e5f6",
      "chunks_indexed": 45,
      "file_type": "pdf",
      "message": "Successfully indexed 45 chunks from file1.pdf"
    },
    {
      "status": "success",
      "filename": "file3.txt",
      "source_id": "g7h8i9j0k1l2",
      "chunks_indexed": 12,
      "file_type": "txt",
      "message": "Successfully indexed 12 chunks from file3.txt"
    }
  ],
  "errors": [
    {
      "filename": "file2.xlsx",
      "error": "Unsupported file type '.xlsx'. Supported: .pdf, .txt, .md, .docx, .json, .csv"
    }
  ],
  "message": "Processed 3 files: 2 succeeded, 1 failed"
}
```

The single-file `POST /upload` endpoint also remains available for backward compatibility.

### 12.2 How It Works — End-to-End Flow

```
User selects 3 files in the browser → Frontend sends multipart POST to /upload/batch
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Backend: /upload/batch endpoint                              │
│                                                               │
│  For EACH file in the batch:                                 │
│    1. Validate filename exists                               │
│    2. Sanitize filename (strip special chars, max 128 chars) │
│    3. Check file extension is supported                      │
│    4. Check file size ≤ MAX_FILE_SIZE_MB                     │
│    5. Save to uploads/ directory                             │
│    6. Run ingest_file(save_path):                            │
│       a. Delete old chunks (if file existed before)          │
│       b. Load → Chunk → Embed → Store in ChromaDB           │
│       c. Load into SQL (if CSV/JSON)                         │
│    7. Record success or error for this file                  │
│                                                               │
│  After ALL files processed:                                  │
│    → Invalidate source cache (once, not per file)            │
│    → Return MultiUploadResponse with per-file results        │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Result: All files indexed into the SAME ChromaDB collection │
│                                                               │
│  ChromaDB collection "rag_documents" now contains:           │
│    • Chunks from file1.pdf (source="file1.pdf")              │
│    • Chunks from file2.csv (source="file2.csv")              │
│    • Chunks from file3.txt (source="file3.txt")              │
│    • Chunks from any previously uploaded files               │
│                                                               │
│  Every chunk has metadata.source = its originating filename  │
└───────────────────────────────────────────────────────────────┘
```

The key backend code in `backend/main.py`:

```python
# backend/main.py — batch upload endpoint
@app.post("/upload/batch", response_model=MultiUploadResponse)
async def upload_files_batch(files: List[UploadFile] = File(...)):
    results = []
    errors = []

    for file in files:
        # Validate, save, and ingest each file individually
        # Failures for one file don't block the others
        try:
            result = ingest_file(save_path)
            results.append(UploadResponse(...))
        except Exception as e:
            errors.append({"filename": safe_name, "error": str(e)})

    # Invalidate cache once after all files processed
    invalidate_source_cache()

    return MultiUploadResponse(
        total_files=len(results) + len(errors),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )
```

**Key design decisions:**
- **Per-file error isolation**: If one file fails (unsupported type, too large, parse error), the other files still get processed successfully.
- **Single cache invalidation**: The source cache is cleared once at the end, not after each file, for efficiency.
- **Same ingestion pipeline**: Each file goes through the exact same `ingest_file()` pipeline as a single upload — including de-duplication, so re-uploading a batch with an existing filename replaces it.

### 12.3 Frontend Multi-File Support

The frontend supports selecting or drag-and-dropping multiple files at once:

- **File input**: `<input type="file" multiple />` — the `multiple` attribute allows selecting several files from the file picker.
- **Drag-and-drop**: The drop zone accepts multiple files dropped simultaneously.
- **Smart routing**: If only 1 file is selected, uses the single `/upload` endpoint. If 2+ files are selected, uses `/upload/batch`.
- **Progress feedback**: Shows total progress (`⏳ Uploading 3 files...`), then a summary (`✅ 3/3 files uploaded (127 total chunks)`).
- **Error reporting**: Per-file errors are displayed via toast notifications so you know exactly which files failed and why.

### 12.4 Cross-File Retrieval — Querying Across All Documents

Once multiple files are uploaded, the agent can search across **all of them simultaneously**:

```
User: "Compare the revenue figures in report.pdf with the data in financials.csv"
    │
    ▼
Agent calls: retrieve_documents(query="revenue figures", source_filter=None)
    │
    ▼
ChromaDB searches ALL chunks from ALL files → returns top-K most relevant
    │
    ▼
Agent may also call: query_database(sql_query="SELECT revenue FROM financials")
    │
    ▼
Agent synthesizes information from both sources into one answer with citations
```

**How cross-file retrieval works:**

| Feature | Behavior |
|---------|----------|
| Default search | `source_filter=None` — searches ALL uploaded files at once |
| Source-filtered search | `source_filter="report.pdf"` — restricts search to one specific file |
| SQL queries | Queries any table created from CSV/JSON uploads |
| Get all chunks | `get_document_chunks(source_filename="report.pdf")` — retrieves all chunks from one file in order |
| Citations | Each citation includes the `source` filename so users know which file the information came from |

The agent's system prompt dynamically lists all available sources and SQL tables, so it always knows what documents and data are available to query.
