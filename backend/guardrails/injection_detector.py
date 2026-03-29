"""
Prompt injection / jailbreak detection.

Uses pattern matching for common attack patterns.
Returns a risk score 0.0–1.0 and a list of triggered rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InjectionResult:
    is_injection: bool
    risk_score: float      # 0.0–1.0
    triggered_rules: List[str]
    sanitised_text: str    # text with injection patterns removed


# ── Detection rules ──────────────────────────────────────────────────────────

_RULES: List[Tuple[str, re.Pattern, float]] = [
    # Rule name, pattern, weight
    (
        "ignore_previous",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|above|prior|instruction|prompt|system)\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "role_switch",
        re.compile(
            r"\b(you are now|pretend (to be|you are)|act as|roleplay as|your new (role|persona|identity))\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "jailbreak_dan",
        re.compile(r"\bDAN\b|\bjailbreak\b|\bunrestricted mode\b", re.IGNORECASE),
        0.95,
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"\b(print|reveal|show|output|display|repeat).{0,20}(system prompt|instructions|context|guidelines)\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "delimiter_injection",
        re.compile(r"(```|---|\[INST\]|<\|im_start\|>|<\|system\|>)", re.IGNORECASE),
        0.6,
    ),
    (
        "base64_payload",
        re.compile(r"(?:[A-Za-z0-9+/]{40,}={1,2})"),
        0.4,
    ),
    (
        "command_execution",
        re.compile(
            r"\b(execute|run|eval|os\.system|subprocess|shell)\b.{0,20}(command|code|script)\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
]

_INJECTION_THRESHOLD = 0.5


def detect_injection(text: str) -> InjectionResult:
    """
    Analyse text for prompt injection patterns.
    """
    triggered: List[str] = []
    max_score = 0.0
    sanitised = text

    for rule_name, pattern, weight in _RULES:
        if pattern.search(text):
            triggered.append(rule_name)
            max_score = max(max_score, weight)
            # Sanitise: replace matched regions with [BLOCKED]
            sanitised = pattern.sub("[BLOCKED]", sanitised)

    is_injection = max_score >= _INJECTION_THRESHOLD

    if is_injection:
        logger.warning(
            "injection_detected",
            rules=triggered,
            score=max_score,
            text_preview=text[:100],
        )

    return InjectionResult(
        is_injection=is_injection,
        risk_score=round(max_score, 3),
        triggered_rules=triggered,
        sanitised_text=sanitised,
    )