"""
Structured JSON logger for agents-gateway.

Every log record is emitted as a single JSON line to stdout so it can be
picked up by any log aggregator (Docker logs, ELK, Loki, etc.).

Usage:
    from logger import get_logger
    log = get_logger(__name__)

    log.info("guardrail.decision", allowed=True, user_text="...")
    log.warning("classifier.out_of_scope", rewritten_query="...")
    log.error("rag.request_failed", error=str(e), file_ids=file_ids)
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }
        }
        if extras:
            base.update(extras)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str, ensure_ascii=False)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class StructuredLogger:
    """Thin wrapper that passes keyword arguments as LogRecord extras."""

    def __init__(self, name: str):
        self._log = _build_logger(name)
        self._log.setLevel(logging.DEBUG)

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, event, stacklevel=3, extra=kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


@contextmanager
def timed(log: StructuredLogger, event: str, **kwargs: Any):
    """Context manager that logs start/end and elapsed_ms."""
    start = time.perf_counter()
    log.debug(f"{event}.start", **kwargs)
    try:
        yield
    except Exception as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        log.error(f"{event}.error", elapsed_ms=elapsed, error=str(exc), **kwargs)
        raise
    else:
        elapsed = int((time.perf_counter() - start) * 1000)
        log.info(f"{event}.done", elapsed_ms=elapsed, **kwargs)
