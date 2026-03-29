"""
LangChain tools available to the agentic RAG system.

Tools:
  1. retrieve_documents  – semantic search in ChromaDB with re-ranking
  2. get_database_schema – introspect SQLite schema
  3. query_database      – execute a natural-language-to-SQL query
  4. request_clarification – signal the LLM wants more info from the user
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.tools import tool

from backend.config import get_settings
from backend.retrieval.sql_store import get_sql_store
from backend.retrieval.vector_store import get_vector_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Re-ranking utilities ─────────────────────────────────────────────────────


def _keyword_overlap_score(query: str, text: str) -> float:
    """Simple keyword overlap between query and chunk text (0-1)."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    overlap = q_words & t_words
    return len(overlap) / len(q_words)


def _reciprocal_rank_fusion(
    semantic_results: list,
    query: str,
    k_constant: int = 60,
) -> list:
    """
    Re-rank results using Reciprocal Rank Fusion (RRF) combining:
      1. Semantic similarity rank (from ChromaDB)
      2. Keyword overlap rank (BM25-like proxy)

    RRF formula: score = sum( 1 / (k + rank_i) ) across all rankers.
    This is a lightweight, dependency-free re-ranking method.
    """
    if not semantic_results:
        return semantic_results

    # Rank 1: Semantic (already sorted by similarity score)
    semantic_ranked = list(range(len(semantic_results)))

    # Rank 2: Keyword overlap
    keyword_scores = [
        _keyword_overlap_score(query, r.get("snippet", ""))
        for r in semantic_results
    ]
    keyword_ranked = sorted(
        range(len(semantic_results)),
        key=lambda i: keyword_scores[i],
        reverse=True,
    )

    # Build RRF scores
    rrf_scores = [0.0] * len(semantic_results)
    for rank, idx in enumerate(semantic_ranked):
        rrf_scores[idx] += 1.0 / (k_constant + rank + 1)
    for rank, idx in enumerate(keyword_ranked):
        rrf_scores[idx] += 1.0 / (k_constant + rank + 1)

    # Sort by RRF score descending
    reranked_indices = sorted(
        range(len(semantic_results)),
        key=lambda i: rrf_scores[i],
        reverse=True,
    )

    reranked = [semantic_results[i] for i in reranked_indices]
    logger.info(
        "reranking_applied",
        method="rrf",
        input_count=len(semantic_results),
        top_rrf_score=round(max(rrf_scores), 4),
    )
    return reranked

# ─── Tool 1: Vector Retrieval ─────────────────────────────────────────────────


@tool
def retrieve_documents(query: str, top_k: int = 10, source_filter: Optional[str] = None) -> str:
    """
    Search the knowledge base (indexed PDFs, DOCX, TXT, CSV, JSON files)
    for text chunks semantically relevant to the query.

    Use this tool whenever the user asks about content from uploaded documents.

    Args:
        query: The search query in natural language.
        top_k: Number of results to return (default 10, max 20).
        source_filter: Optional filename to restrict search to a specific document.
                       Use this when the user asks about a particular file.
                       Pass the exact filename (e.g. "report.pdf", "data.csv").
                       Leave empty/None to search across ALL documents.

    Returns:
        JSON string containing:
          - results: list of chunks
          - citations: same list (compat key used by many agent graphs)
          - count: number of chunks
          - message: optional status message
          - error: optional error message
    """
    try:
        settings = get_settings()
    except Exception as e:
        logger.error("tool_retrieve_documents_settings_error", error=str(e))
        return json.dumps(
            {"results": [], "citations": [], "count": 0, "error": f"Settings error: {e}"},
            ensure_ascii=False,
        )

    try:
        # Normalize and clamp top_k
        try:
            top_k_int = int(top_k)
        except Exception:
            top_k_int = 5

        top_k_int = min(max(1, top_k_int), 20)
        actual_k = min(top_k_int, getattr(settings, "retrieval_top_k", top_k_int))

        query_str = (query or "").strip()
        if not query_str:
            return json.dumps(
                {
                    "results": [],
                    "citations": [],
                    "count": 0,
                    "message": "Empty query. Provide a search query.",
                },
                ensure_ascii=False,
            )

        logger.info("tool_retrieve_documents", query=query_str[:80], top_k=actual_k)

        # Init vector store (LangChain Chroma)
        vs = get_vector_store()

        # Count docs — handle both raw Chroma and ChromaVectorStoreAdapter
        doc_count = 0
        try:
            store = getattr(vs, "_store", vs)
            if hasattr(store, "_collection") and store._collection is not None:
                doc_count = int(store._collection.count())
        except Exception as e:
            logger.error("tool_retrieve_documents_vs_count_error", error=str(e))
            doc_count = 0

        if doc_count == 0:
            return json.dumps(
                {"results": [], "citations": [], "count": 0, "message": "No documents indexed yet."},
                ensure_ascii=False,
            )

        # Retrieve (LangChain Chroma does embedding internally)
        # Build optional metadata filter for source-specific queries
        chroma_filter = None
        if source_filter:
            sf = str(source_filter).strip()
            if sf:
                chroma_filter = {"source": sf}
                logger.info("tool_retrieve_documents_source_filter", source=sf)

        try:
            docs_and_scores = vs.similarity_search_with_score(query_str, k=actual_k, filter=chroma_filter)
        except Exception as e:
            logger.error("tool_retrieve_documents_search_error", error=str(e))
            return json.dumps(
                {"results": [], "citations": [], "count": 0, "error": f"Vector search failed: {e}"},
                ensure_ascii=False,
            )

        # Build payload
        results = []
        for doc, raw_score in docs_and_scores or []:
            meta = getattr(doc, "metadata", None) or {}

            # Normalise ChromaDB distance → similarity (0–1).
            # ChromaDB returns L2 distance by default; lower = better.
            # Cosine distance is in [0, 2]; convert to similarity.
            try:
                dist = float(raw_score)
                # If store returns cosine distance (0–2): sim = 1 - dist/2
                # If store returns L2 distance: clamp then convert
                if dist < 0:
                    score = 0.0
                elif dist <= 2.0:
                    score = round(max(0.0, 1.0 - dist / 2.0), 4)
                else:
                    score = round(max(0.0, 1.0 / (1.0 + dist)), 4)
            except (TypeError, ValueError):
                score = 0.0

            results.append(
                {
                    "source": meta.get("source")
                    or meta.get("file_path")
                    or meta.get("filename")
                    or "unknown",
                    "chunk_id": meta.get("chunk_id") or meta.get("id") or "?",
                    "page_number": meta.get("page") or meta.get("page_number"),
                    "score": score,
                    "snippet": getattr(doc, "page_content", "") or "",
                }
            )

        # Re-rank results using RRF (semantic + keyword fusion)
        if len(results) > 1:
            results = _reciprocal_rank_fusion(results, query_str)

        # ── Keyword presence check ────────────────────────────────────
        # Extract key terms from the query and check if they appear in
        # any retrieved chunk. If not, warn the LLM that the results
        # may be about a DIFFERENT (but semantically similar) topic.
        query_term_match = True
        query_term_warning = ""
        _stop_words = {
            "what", "is", "are", "was", "were", "do", "does", "did",
            "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "not", "by", "from", "with", "about",
            "how", "why", "when", "where", "who", "which", "can",
            "you", "me", "my", "your", "i", "we", "they", "it",
            "this", "that", "tell", "explain", "describe", "mean",
            "please", "could", "would", "should", "will", "shall",
            "be", "been", "being", "have", "has", "had", "may",
            "might", "must", "need", "use", "used", "using",
            "say", "says", "said", "does", "work", "works",
            "according", "document", "project", "system", "role",
        }

        # Split query into individual word stems, also splitting hyphenated words
        raw_words = query_str.lower().split()
        key_terms = []
        for w in raw_words:
            w_clean = w.strip("?.,!'\"()")
            if w_clean not in _stop_words and len(w_clean) > 2:
                key_terms.append(w_clean)
                # Also add sub-words for hyphenated terms (e.g. "fine-tuning" -> "fine", "tuning")
                if "-" in w_clean:
                    for part in w_clean.split("-"):
                        if part not in _stop_words and len(part) > 2:
                            key_terms.append(part)

        if key_terms and results:
            all_text = " ".join(r.get("snippet", "") for r in results).lower()

            def _term_found(term: str, text: str) -> bool:
                """Check if term (or a close stem variant) appears in text."""
                if term in text:
                    return True
                # Check stem variants: "hashing" -> "hash", "routing" -> "rout"
                if len(term) > 4:
                    stem = term[:max(4, len(term) - 3)]
                    if stem in text:
                        return True
                return False

            missing_terms = [t for t in key_terms if not _term_found(t, all_text)]
            # Dedupe missing terms
            missing_terms = list(dict.fromkeys(missing_terms))
            missing_ratio = len(missing_terms) / len(key_terms) if key_terms else 0

            if missing_terms and missing_ratio >= 0.6:
                # Most key terms missing — likely a completely different topic
                query_term_match = False
                query_term_warning = (
                    f"STOP — TOPIC MISMATCH DETECTED: The key query terms {missing_terms} "
                    f"({len(missing_terms)}/{len(key_terms)} key terms missing) "
                    f"do NOT appear in ANY retrieved chunk. The retrieved results are about "
                    f"a DIFFERENT topic. You MUST NOT use these results to answer the question. "
                    f"Instead, tell the user: 'I couldn't find information about [their topic] "
                    f"in the uploaded documents.'"
                )
            elif missing_terms and missing_ratio > 0:
                # Some terms missing but most found — soft warning, allow usage
                query_term_warning = (
                    f"NOTE: Some query terms {missing_terms} were not found verbatim in the "
                    f"retrieved chunks, but most key terms matched. The chunks are likely relevant. "
                    f"Use the retrieved content to answer the question."
                )
                logger.info(
                    "query_term_mismatch",
                    query=query_str[:80],
                    missing_terms=missing_terms,
                )

        payload = {
            "results": results,
            "citations": results,  # compat key
            "count": len(results),
            "query_term_match": query_term_match,
        }

        if query_term_warning:
            payload["query_term_warning"] = query_term_warning

        if len(results) == 0:
            payload["message"] = "No relevant chunks found."

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.error("tool_retrieve_documents_unhandled_error", error=str(e))
        return json.dumps(
            {"results": [], "citations": [], "count": 0, "error": f"Unhandled retrieve_documents error: {e}"},
            ensure_ascii=False,
        )


# ─── Tool 2: Database Schema ──────────────────────────────────────────────────


@tool
def get_database_schema() -> str:
    """
    Retrieve the schema of all tables in the structured database (SQLite).

    Call this BEFORE query_database to understand what tables and columns
    are available. Returns table names, column names, and data types.
    """
    try:
        logger.info("tool_get_database_schema")
        sql_store = get_sql_store()
        schema_text = sql_store.get_schema_as_text()
        tables = sql_store.list_tables()
        return json.dumps({"schema_description": schema_text, "tables": tables}, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_get_database_schema_error", error=str(e))
        return json.dumps({"error": str(e), "schema_description": "", "tables": []}, ensure_ascii=False)


# ─── Tool 3: Database Query ───────────────────────────────────────────────────


@tool
def query_database(sql_query: str) -> str:
    """
    Execute a SQL SELECT query against the structured database (SQLite).

    IMPORTANT:
    - Only SELECT queries are allowed.
    - Always call get_database_schema first to know the table/column names.
    - Use standard SQLite syntax.
    - Limit results to avoid overwhelming output (e.g., LIMIT 20).

    Args:
        sql_query: A valid SQLite SELECT statement.

    Returns:
        JSON string with 'rows' (list of dicts) and 'row_count'.
        If there is an error, returns an 'error' field instead.
    """
    try:
        sql_query = (sql_query or "").strip()
        logger.info("tool_query_database", sql=sql_query[:120])

        if not sql_query:
            return json.dumps({"error": "Empty SQL query."}, ensure_ascii=False)

        # Simple guardrail: allow only SELECT
        if not sql_query.lower().lstrip().startswith("select"):
            return json.dumps(
                {"error": "Only SELECT queries are allowed.", "sql": sql_query},
                ensure_ascii=False,
            )

        sql_store = get_sql_store()
        rows, error = sql_store.execute_query(sql_query)

        if error:
            return json.dumps({"error": error, "sql": sql_query}, ensure_ascii=False)

        truncated = False
        if len(rows) > 50:
            rows = rows[:50]
            truncated = True

        return json.dumps({"rows": rows, "row_count": len(rows), "truncated": truncated}, ensure_ascii=False)

    except Exception as e:
        logger.error("tool_query_database_error", error=str(e))
        return json.dumps({"error": str(e), "sql": sql_query}, ensure_ascii=False)


# ─── Tool 4: Request Clarification ───────────────────────────────────────────


@tool
def request_clarification(question_for_user: str) -> str:
    """
    Use this tool when the user's question is ambiguous and you need
    more information to give a correct answer.

    Ask ONE clear, specific clarifying question.

    Args:
        question_for_user: The clarifying question to ask the user.

    Returns:
        A marker string; the agent should then end its turn and
        present the clarifying question to the user.
    """
    try:
        question_for_user = (question_for_user or "").strip()
        logger.info("tool_request_clarification", question=question_for_user[:100])
        return json.dumps({"clarification_needed": True, "question": question_for_user}, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_request_clarification_error", error=str(e))
        return json.dumps(
            {"clarification_needed": True, "question": "Could you clarify your request?", "error": str(e)},
            ensure_ascii=False,
        )


# ─── Tool 5: Get All Document Chunks ──────────────────────────────────────────


@tool
def get_document_chunks(source_filename: str, max_chunks: int = 150) -> str:
    """
    Retrieve ALL chunks from a specific document by its exact filename.

    USE THIS TOOL (not retrieve_documents) when:
    - The user asks to "summarize", "tell me about", or "what's in" a specific file
    - The user says "tell me about the uploaded file" or "summarize the document"
    - You need a COMPLETE overview of a document's content, not just search results

    This returns chunks in their original order, giving you the full document content.
    Unlike retrieve_documents (which does semantic search and may miss content),
    this tool returns ALL chunks so you can provide a comprehensive summary.

    Args:
        source_filename: The exact filename (e.g., "report.pdf", "data.csv").
                         Must match one of the indexed document filenames.
        max_chunks: Maximum number of chunks to return (default 150, max 300).

    Returns:
        JSON string with all chunks from the document, ordered by position.
    """
    try:
        source_filename = (source_filename or "").strip()
        if not source_filename:
            return json.dumps(
                {"results": [], "citations": [], "count": 0, "error": "No filename provided."},
                ensure_ascii=False,
            )

        max_chunks = min(max(1, int(max_chunks)), 300)
        logger.info("tool_get_document_chunks", source=source_filename, max_chunks=max_chunks)

        vs = get_vector_store()
        store = getattr(vs, "_store", vs)
        col = getattr(store, "_collection", None)

        if col is None:
            return json.dumps(
                {"results": [], "citations": [], "count": 0, "error": "Vector store not available."},
                ensure_ascii=False,
            )

        # Get all chunks for this source
        data = col.get(
            where={"source": source_filename},
            include=["documents", "metadatas"],
        )

        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        if not docs:
            return json.dumps(
                {"results": [], "citations": [], "count": 0,
                 "message": f"No chunks found for '{source_filename}'. Check the filename."},
                ensure_ascii=False,
            )

        # Build results sorted by chunk index (preserves document order)
        items = []
        for doc_text, meta in zip(docs, metas):
            if not isinstance(meta, dict):
                meta = {}
            items.append({
                "source": meta.get("source", source_filename),
                "chunk_id": meta.get("chunk_id", ""),
                "page_number": meta.get("page_number"),
                "global_chunk_index": meta.get("global_chunk_index", 0),
                "score": 1.0,  # direct retrieval = full relevance
                "snippet": doc_text or "",
            })

        # Sort by global_chunk_index to preserve document order
        items.sort(key=lambda x: x.get("global_chunk_index", 0))

        # Truncate to max_chunks
        items = items[:max_chunks]

        # Build compact results (avoid duplicating into both "results" and "citations")
        # Use full chunk text — chunks are typically ~512 chars so no need to truncate aggressively.
        # Only truncate truly oversized chunks to avoid context window overflow.
        MAX_SNIPPET_CHARS = 1200
        compact_items = []
        for it in items:
            snippet = it["snippet"]
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS] + "…"
            compact_items.append({
                "source": it["source"],
                "chunk_id": it["chunk_id"],
                "page_number": it.get("page_number"),
                "score": it["score"],
                "snippet": snippet,
            })

        return json.dumps(
            {"results": compact_items, "count": len(compact_items),
             "message": f"Retrieved {len(compact_items)} of {len(docs)} total chunks for '{source_filename}'."},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error("tool_get_document_chunks_error", error=str(e))
        return json.dumps(
            {"results": [], "citations": [], "count": 0, "error": f"Error: {e}"},
            ensure_ascii=False,
        )


# ─── Tool registry ────────────────────────────────────────────────────────────

ALL_TOOLS = [
    retrieve_documents,
    get_document_chunks,
    get_database_schema,
    query_database,
    request_clarification,
]