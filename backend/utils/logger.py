"""
Structured logging via structlog with request tracing support.
"""
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for request tracing
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str | None = None) -> str:
    """Set a request ID for the current context. Generates one if not provided."""
    rid = request_id or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """Get the current request ID."""
    return _request_id.get("")


def add_request_id(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Add request_id to every log entry for tracing."""
    rid = _request_id.get("")
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def add_severity(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Map structlog level to GCP-compatible severity."""
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
    }
    event_dict["severity"] = level_map.get(method, "INFO")
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            add_request_id,
            add_severity,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> Any:
    return structlog.get_logger(name)