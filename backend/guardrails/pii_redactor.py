"""
PII detection and redaction.

Uses regex patterns for common PII types.
If microsoft/presidio is installed, provides deeper entity detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    entity_type: str
    original: str
    redacted: str
    start: int
    end: int


# ── Regex patterns ───────────────────────────────────────────────────────────

_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL]",
    ),
    (
        "PHONE_US",
        re.compile(
            r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "[PHONE]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "[CREDIT_CARD]",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "[IP_ADDRESS]",
    ),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"\b(?:dob|date of birth|born on)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "[DOB]",
    ),
]


class PIIRedactor:
    def __init__(self, use_presidio: bool = False):
        self._use_presidio = use_presidio
        self._presidio_analyzer = None
        self._presidio_anonymizer = None

        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._presidio_analyzer = AnalyzerEngine()
                self._presidio_anonymizer = AnonymizerEngine()
                logger.info("pii_redactor_mode", mode="presidio")
            except ImportError:
                logger.warning(
                    "presidio_unavailable",
                    msg="Falling back to regex PII detection",
                )
                self._use_presidio = False

        if not self._use_presidio:
            logger.info("pii_redactor_mode", mode="regex")

    def detect_and_redact(self, text: str) -> Tuple[str, List[PIIMatch], bool]:
        """
        Returns (redacted_text, matches, pii_found).
        """
        if self._use_presidio and self._presidio_analyzer:
            return self._presidio_redact(text)
        return self._regex_redact(text)

    def _regex_redact(
        self, text: str
    ) -> Tuple[str, List[PIIMatch], bool]:
        matches: List[PIIMatch] = []
        redacted = text

        for entity_type, pattern, placeholder in _PATTERNS:
            for m in pattern.finditer(redacted):
                matches.append(
                    PIIMatch(
                        entity_type=entity_type,
                        original=m.group(),
                        redacted=placeholder,
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Apply replacements (iterate in reverse to preserve offsets)
        for entity_type, pattern, placeholder in _PATTERNS:
            redacted = pattern.sub(placeholder, redacted)

        return redacted, matches, len(matches) > 0

    def _presidio_redact(
        self, text: str
    ) -> Tuple[str, List[PIIMatch], bool]:
        results = self._presidio_analyzer.analyze(text=text, language="en")
        if not results:
            return text, [], False

        anonymized = self._presidio_anonymizer.anonymize(
            text=text, analyzer_results=results
        )
        matches = [
            PIIMatch(
                entity_type=r.entity_type,
                original=text[r.start : r.end],
                redacted=f"[{r.entity_type}]",
                start=r.start,
                end=r.end,
            )
            for r in results
        ]
        return anonymized.text, matches, True