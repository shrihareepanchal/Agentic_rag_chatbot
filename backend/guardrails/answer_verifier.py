"""
Answer Verifier — Auto-corrects the 3 fundamental LLM flaws:

  Flaw 1  Knowledge Cutoff  → Detects when the LLM answers from training data
                               instead of retrieved documents.
  Flaw 2  Hallucination     → Detects fabricated facts (numbers, names, dates,
                               URLs) that don't appear in any retrieved chunk.
  Flaw 3  No Source Attribution → Detects when the answer lacks citations and
                                   auto-injects them from the retrieved chunks.

Each flaw has a dedicated detector + corrector.  The main entry point is
``verify_and_correct()``, called from the output_guard_node in graph.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VerificationIssue:
    """A single detected issue in the LLM answer."""
    flaw_type: str          # "knowledge_cutoff" | "hallucination" | "no_attribution"
    severity: str           # "high" | "medium" | "low"
    description: str        # Human-readable explanation
    evidence: str = ""      # The problematic text from the answer


@dataclass
class VerificationResult:
    """Result of verifying an LLM answer against retrieved chunks."""
    is_clean: bool                              # True if no issues found
    corrected_answer: str                       # Auto-corrected answer text
    issues: List[VerificationIssue] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    missing_citations: List[Dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Flaw 1 — Knowledge Cutoff Detector
# ──────────────────────────────────────────────────────────────────────────────

# Phrases that strongly indicate the LLM is using training data, not documents
_TRAINING_DATA_PHRASES = [
    re.compile(r"\b(as of my (last |knowledge |training )?(update|cutoff|knowledge))", re.I),
    re.compile(r"\b(my training data|my training knowledge)\b", re.I),
    re.compile(r"\b(as an AI( language model)?|as a large language model)\b", re.I),
    re.compile(r"\bgenerally speaking\b.*\b(in most cases|typically)\b", re.I),
    re.compile(r"\b(based on (my|general) knowledge)\b", re.I),
    re.compile(r"\b(according to (publicly available|general|common) (information|knowledge))\b", re.I),
    re.compile(r"\b(I don't have access to (real-time|current|live|your))\b", re.I),
    re.compile(r"\b(in the (field|domain|area) of)\b.*\bgenerally\b", re.I),
    re.compile(r"\b(this is a (common|well-known|widely accepted))\b", re.I),
]

# Phrases that signal the LLM is inventing knowledge beyond retrieved content
_EXTERNAL_KNOWLEDGE_PHRASES = [
    re.compile(r"\b(it is (widely|commonly|generally) (known|accepted|believed))\b", re.I),
    re.compile(r"\b(research (has shown|suggests|indicates))\b", re.I),
    re.compile(r"\b(studies (have shown|suggest|indicate))\b", re.I),
    re.compile(r"\b(experts (say|believe|recommend|suggest))\b", re.I),
    re.compile(r"\b(according to (industry|recent) (standards|reports|studies))\b", re.I),
    re.compile(r"\b(in (practice|reality|real world))\b.*\b(usually|typically|often)\b", re.I),
]


def detect_knowledge_cutoff(answer: str, has_citations: bool) -> List[VerificationIssue]:
    """
    Detect when the LLM is answering from its training data instead of
    from the retrieved documents.

    Checks for:
    - Explicit training-data disclaimers ("As of my last update...")
    - External knowledge phrases ("Research has shown...")
    - Answers without any retrieval citations that contain factual claims
    """
    issues: List[VerificationIssue] = []

    # Check for explicit training data phrases
    for pattern in _TRAINING_DATA_PHRASES:
        match = pattern.search(answer)
        if match:
            issues.append(VerificationIssue(
                flaw_type="knowledge_cutoff",
                severity="high",
                description="LLM is using its training data instead of retrieved documents.",
                evidence=match.group(0),
            ))

    # Check for external knowledge phrases
    for pattern in _EXTERNAL_KNOWLEDGE_PHRASES:
        match = pattern.search(answer)
        if match:
            issues.append(VerificationIssue(
                flaw_type="knowledge_cutoff",
                severity="medium",
                description="LLM appears to be injecting general knowledge not from documents.",
                evidence=match.group(0),
            ))

    # If the answer contains factual claims but NO citations at all,
    # and it's not a greeting or "I don't know" response, flag it
    _factual_patterns = re.compile(
        r"(\d{4})|(\$[\d,.]+)|(\d+%)|(\d+\.\d+)|"
        r"(according to|the report|the document|the data)",
        re.I,
    )
    _no_info_patterns = re.compile(
        r"(couldn't find|don't have|no information|not found|"
        r"no relevant|unable to find|not available|no documents)",
        re.I,
    )
    _greeting_patterns = re.compile(
        r"^(hi|hello|hey|good |welcome|how can I|I'm here to help)",
        re.I,
    )

    if (not has_citations
            and _factual_patterns.search(answer)
            and not _no_info_patterns.search(answer)
            and not _greeting_patterns.search(answer)):
        issues.append(VerificationIssue(
            flaw_type="knowledge_cutoff",
            severity="medium",
            description="Answer contains factual claims but no retrieved citations — "
                        "may be using training data.",
            evidence="(no citations found for factual answer)",
        ))

    return issues


def correct_knowledge_cutoff(answer: str, issues: List[VerificationIssue]) -> Tuple[str, List[str]]:
    """
    Auto-correct knowledge cutoff issues:
    - Remove training-data disclaimers
    - Replace external-knowledge phrases with grounded language
    - Append a transparency notice if the answer may be from training data
    """
    corrected = answer
    corrections: List[str] = []

    # Remove explicit training data phrases
    for pattern in _TRAINING_DATA_PHRASES:
        if pattern.search(corrected):
            corrected = pattern.sub("", corrected)
            corrections.append("Removed training-data disclaimer")

    # Replace external knowledge phrases with hedged language
    for pattern in _EXTERNAL_KNOWLEDGE_PHRASES:
        match = pattern.search(corrected)
        if match:
            corrected = pattern.sub("Based on the uploaded documents", corrected)
            corrections.append(f"Replaced external knowledge phrase: '{match.group(0)}'")

    # Clean up leftover double spaces / empty lines
    corrected = re.sub(r"  +", " ", corrected)
    corrected = re.sub(r"\n{3,}", "\n\n", corrected)

    # If high-severity issues remain (no citations for factual claims),
    # append transparency disclaimer
    high_severity = any(i.severity == "high" for i in issues)
    no_citation_issue = any("no citations" in i.evidence for i in issues)

    if high_severity or no_citation_issue:
        disclaimer = (
            "\n\n---\n"
            "⚠️ **Auto-Correction Notice:** Parts of this response could not be "
            "verified against the uploaded documents. For the most accurate answers, "
            "please upload relevant documents and ask again."
        )
        if disclaimer.strip() not in corrected:
            corrected = corrected.rstrip() + disclaimer
            corrections.append("Added transparency disclaimer for unverified content")

    return corrected.strip(), corrections


# ──────────────────────────────────────────────────────────────────────────────
# Flaw 2 — Hallucination Detector
# ──────────────────────────────────────────────────────────────────────────────

# Extract specific factual claims from text
_NUMBER_PATTERN = re.compile(r"(?<!\w)(\$?\d[\d,]*\.?\d*\s*(?:%|percent|billion|million|thousand|k|m|b)?)\b", re.I)
_DATE_PATTERN = re.compile(
    r"\b("
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s*\d{4}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2}"
    r"|Q[1-4]\s*(?:19|20)\d{2}"
    r")\b",
    re.I,
)
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'>]+", re.I)
_PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _extract_facts(text: str) -> Dict[str, List[str]]:
    """Extract verifiable facts (numbers, dates, URLs, proper nouns) from text."""
    return {
        "numbers": list(set(m.group(1).strip() for m in _NUMBER_PATTERN.finditer(text))),
        "dates": list(set(m.group(1).strip() for m in _DATE_PATTERN.finditer(text))),
        "urls": list(set(m.group(0).strip() for m in _URL_PATTERN.finditer(text))),
        "proper_nouns": list(set(m.group(1).strip() for m in _PROPER_NOUN_PATTERN.finditer(text))),
    }


def _normalize(text: str) -> str:
    """Normalize text for comparison — lowercase, remove punctuation and extra spaces."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def detect_hallucination(
    answer: str,
    chunk_texts: List[str],
) -> List[VerificationIssue]:
    """
    Detect fabricated facts in the LLM answer that don't appear in any
    retrieved chunk.

    Checks:
    - Numbers/statistics not traceable to any chunk
    - Dates not found in any chunk
    - URLs invented by the LLM (not in any chunk)
    - Proper nouns not present in source material
    """
    if not chunk_texts or not answer.strip():
        return []

    issues: List[VerificationIssue] = []

    # Combine all chunk text for searching
    all_chunks_text = " ".join(chunk_texts)
    chunks_normalized = _normalize(all_chunks_text)

    # Extract facts from the answer
    answer_facts = _extract_facts(answer)

    # Check numbers — are they in the chunks?
    for num in answer_facts["numbers"]:
        num_clean = num.replace(",", "").replace("$", "").strip()
        # A number must appear somewhere in the chunks
        if num_clean and len(num_clean) >= 2 and num_clean not in chunks_normalized:
            # Also try the formatted version
            if num.replace(",", "").replace("$", "").lower() not in chunks_normalized:
                issues.append(VerificationIssue(
                    flaw_type="hallucination",
                    severity="high",
                    description=f"Number '{num}' not found in any retrieved document chunk.",
                    evidence=num,
                ))

    # Check dates — are they in the chunks?
    for date_str in answer_facts["dates"]:
        date_normalized = _normalize(date_str)
        if date_normalized not in chunks_normalized:
            issues.append(VerificationIssue(
                flaw_type="hallucination",
                severity="high",
                description=f"Date '{date_str}' not found in any retrieved document chunk.",
                evidence=date_str,
            ))

    # Check URLs — this is a critical hallucination vector
    for url in answer_facts["urls"]:
        url_lower = url.lower().rstrip("/")
        if url_lower not in all_chunks_text.lower():
            issues.append(VerificationIssue(
                flaw_type="hallucination",
                severity="high",
                description=f"URL '{url}' was fabricated — not found in any document.",
                evidence=url,
            ))

    # Check proper nouns (only flag if multiple words, e.g. "John Smith")
    # Only flag if the noun is significant (not common English phrases)
    _common_phrases = {
        "united states", "new york", "los angeles", "san francisco",
        "would you", "could you", "based on", "according to",
        "more about", "key points", "main findings", "next steps",
    }
    for noun in answer_facts["proper_nouns"]:
        noun_normalized = _normalize(noun)
        if (noun_normalized not in chunks_normalized
                and noun_normalized not in _common_phrases
                and len(noun) > 5):
            issues.append(VerificationIssue(
                flaw_type="hallucination",
                severity="medium",
                description=f"Proper noun '{noun}' not found in any retrieved chunk.",
                evidence=noun,
            ))

    return issues


def correct_hallucination(
    answer: str,
    issues: List[VerificationIssue],
    chunk_texts: List[str],
) -> Tuple[str, List[str]]:
    """
    Auto-correct hallucinated content:
    - Strike fabricated URLs (replace with warning)
    - Mark unverified numbers/dates with a footnote
    - Add a hallucination warning block if issues are severe
    """
    corrected = answer
    corrections: List[str] = []

    # Replace fabricated URLs with a warning
    url_issues = [i for i in issues if i.flaw_type == "hallucination" and "URL" in i.description]
    for issue in url_issues:
        corrected = corrected.replace(
            issue.evidence,
            f"~~{issue.evidence}~~ *(link not found in documents)*",
        )
        corrections.append(f"Struck fabricated URL: {issue.evidence}")

    # Count high-severity hallucinations
    high_hallucinations = [
        i for i in issues
        if i.flaw_type == "hallucination" and i.severity == "high"
    ]

    if len(high_hallucinations) >= 3:
        # Severe hallucination — many fabricated facts
        warning = (
            "\n\n---\n"
            "⚠️ **Hallucination Warning:** This response contains "
            f"**{len(high_hallucinations)} facts** that could not be verified "
            "against the uploaded documents (numbers, dates, or URLs). "
            "These may have been generated from the model's training data "
            "rather than your documents. Please verify critical information "
            "against the original source."
        )
        corrected = corrected.rstrip() + warning
        corrections.append(
            f"Added hallucination warning ({len(high_hallucinations)} unverified facts)"
        )
    elif high_hallucinations:
        # Moderate — a few unverified facts
        unverified_items = ", ".join(i.evidence for i in high_hallucinations[:5])
        note = (
            "\n\n---\n"
            f"ℹ️ **Note:** The following items could not be verified in the uploaded "
            f"documents and may be approximate: {unverified_items}"
        )
        corrected = corrected.rstrip() + note
        corrections.append(f"Added verification note for: {unverified_items}")

    return corrected.strip(), corrections


# ──────────────────────────────────────────────────────────────────────────────
# Flaw 3 — Source Attribution Detector
# ──────────────────────────────────────────────────────────────────────────────

def detect_missing_attribution(
    answer: str,
    citations: List[Dict],
) -> List[VerificationIssue]:
    """
    Detect when the answer lacks proper source attribution.

    An answer needs attribution when:
    - It contains factual claims but no citations exist
    - Citations exist but the answer doesn't reference source documents
    - The answer appears substantive (not a greeting or "I don't know")
    """
    issues: List[VerificationIssue] = []

    # Skip attribution checks for non-factual responses
    _skip_patterns = re.compile(
        r"^(hi|hello|hey|good |welcome|how can I|I'm here|"
        r"couldn't find|don't have|no information|not found|"
        r"no relevant|unable to find|❓|I was unable)",
        re.I,
    )
    if _skip_patterns.search(answer.strip()):
        return issues

    # Check: answer is substantial (>100 chars of factual content) but NO citations
    if len(answer) > 100 and not citations:
        # Check if it's actually making claims (not just asking for clarification)
        has_factual_content = bool(re.search(
            r"(\d+|according|report|document|data|information|found|shows)",
            answer, re.I,
        ))
        if has_factual_content:
            issues.append(VerificationIssue(
                flaw_type="no_attribution",
                severity="high",
                description="Substantive answer with factual claims but no source citations.",
                evidence="(no citations in response)",
            ))

    # Check: citations exist but answer doesn't mention ANY source
    if citations:
        source_names = set(c.get("source", "") for c in citations if c.get("source"))
        answer_lower = answer.lower()
        any_source_mentioned = any(
            src.lower().replace("_", " ").split(".")[0] in answer_lower
            for src in source_names
        )
        if not any_source_mentioned and len(answer) > 150:
            issues.append(VerificationIssue(
                flaw_type="no_attribution",
                severity="low",
                description="Answer does not explicitly reference source documents "
                            "despite having citations available.",
                evidence=", ".join(source_names),
            ))

    return issues


def correct_missing_attribution(
    answer: str,
    issues: List[VerificationIssue],
    citations: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Auto-correct missing attribution:
    - Append a "Sources" section if citations exist but aren't referenced
    - Add a disclaimer if answer has facts but no citations at all
    """
    corrected = answer
    corrections: List[str] = []

    # Get unique source names from citations
    unique_sources: List[str] = []
    seen = set()
    for c in citations:
        src = c.get("source", "")
        if src and src not in seen:
            seen.add(src)
            unique_sources.append(src)

    # If citations exist, append a Sources section
    if unique_sources and any(i.flaw_type == "no_attribution" for i in issues):
        sources_block = "\n\n---\n📄 **Sources:**\n"
        for src in unique_sources[:5]:
            # Find the best score for this source
            best_score = max(
                (c.get("score", 0) for c in citations if c.get("source") == src),
                default=0,
            )
            page_numbers = sorted(set(
                c.get("page_number")
                for c in citations
                if c.get("source") == src and c.get("page_number")
            ))
            page_str = f" (Pages: {', '.join(str(p) for p in page_numbers)})" if page_numbers else ""
            sources_block += f"- **{src}**{page_str} — relevance: {best_score:.0%}\n"

        # Only add if not already present
        if "📄 **Sources:**" not in corrected:
            corrected = corrected.rstrip() + sources_block
            corrections.append(f"Added source attribution: {', '.join(unique_sources[:5])}")

    # If NO citations at all but answer has factual claims, add disclaimer
    if not citations and any(
        i.flaw_type == "no_attribution" and i.severity == "high" for i in issues
    ):
        disclaimer = (
            "\n\n---\n"
            "⚠️ **No Source Attribution:** This response could not be linked "
            "to any specific uploaded document. The information may come from "
            "the model's general knowledge. Upload relevant documents for "
            "verified, cited answers."
        )
        if "No Source Attribution" not in corrected:
            corrected = corrected.rstrip() + disclaimer
            corrections.append("Added no-attribution disclaimer")

    return corrected.strip(), corrections


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point — Run All 3 Checks
# ──────────────────────────────────────────────────────────────────────────────


def verify_and_correct(
    answer: str,
    citations: List[Dict],
    chunk_texts: Optional[List[str]] = None,
) -> VerificationResult:
    """
    Run all three flaw detectors and auto-correct the LLM answer.

    This is the main function called from output_guard_node in graph.py.

    Args:
        answer:      The LLM's generated answer text.
        citations:   List of citation dicts from retrieve_documents results.
        chunk_texts: List of raw chunk text snippets (for hallucination check).
                     If None, extracted from citations[].snippet.

    Returns:
        VerificationResult with corrected answer, issues found, and
        corrections applied.
    """
    if not answer or not answer.strip():
        return VerificationResult(is_clean=True, corrected_answer=answer)

    # Build chunk texts from citation snippets if not provided
    if chunk_texts is None:
        chunk_texts = [
            c.get("snippet", "") for c in citations
            if c.get("snippet", "").strip()
        ]

    all_issues: List[VerificationIssue] = []
    all_corrections: List[str] = []
    corrected = answer

    # ── Flaw 1: Knowledge Cutoff ─────────────────────────────────────────
    cutoff_issues = detect_knowledge_cutoff(corrected, has_citations=bool(citations))
    if cutoff_issues:
        all_issues.extend(cutoff_issues)
        corrected, corrections = correct_knowledge_cutoff(corrected, cutoff_issues)
        all_corrections.extend(corrections)
        logger.warning(
            "flaw_1_knowledge_cutoff_detected",
            issue_count=len(cutoff_issues),
            corrections=corrections,
        )

    # ── Flaw 2: Hallucination ────────────────────────────────────────────
    hallucination_issues = detect_hallucination(corrected, chunk_texts)
    if hallucination_issues:
        all_issues.extend(hallucination_issues)
        corrected, corrections = correct_hallucination(
            corrected, hallucination_issues, chunk_texts,
        )
        all_corrections.extend(corrections)
        logger.warning(
            "flaw_2_hallucination_detected",
            issue_count=len(hallucination_issues),
            types=[i.evidence for i in hallucination_issues[:5]],
            corrections=corrections,
        )

    # ── Flaw 3: No Source Attribution ────────────────────────────────────
    attribution_issues = detect_missing_attribution(corrected, citations)
    if attribution_issues:
        all_issues.extend(attribution_issues)
        corrected, corrections = correct_missing_attribution(
            corrected, attribution_issues, citations,
        )
        all_corrections.extend(corrections)
        logger.warning(
            "flaw_3_no_attribution_detected",
            issue_count=len(attribution_issues),
            corrections=corrections,
        )

    is_clean = len(all_issues) == 0

    if not is_clean:
        logger.info(
            "answer_verification_complete",
            total_issues=len(all_issues),
            total_corrections=len(all_corrections),
            flaw_types=list(set(i.flaw_type for i in all_issues)),
        )

    return VerificationResult(
        is_clean=is_clean,
        corrected_answer=corrected,
        issues=all_issues,
        corrections_applied=all_corrections,
        missing_citations=[],
    )
