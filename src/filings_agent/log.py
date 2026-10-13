"""Structured logging with a per-run correlation ID."""

from __future__ import annotations

import logging
import sys
import uuid

import structlog


def configure_logging(*, json: bool | None = None, level: int = logging.INFO) -> None:
    """Configure structlog. JSON when not attached to a TTY (CI, Docker), pretty otherwise."""
    if json is None:
        json = not sys.stderr.isatty()
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def new_run_id() -> str:
    """Generate a run ID and bind it to every subsequent log line in this context."""
    run_id = uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return run_id
