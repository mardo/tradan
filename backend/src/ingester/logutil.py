"""
Central logging for the ingester. Configure once per process.

Environment:
  INGEST_LOG_LEVEL   DEBUG, INFO, WARNING, ERROR (default: INFO)

Share logs from a run with:
  INGEST_LOG_LEVEL=DEBUG PYTHONUNBUFFERED=1 uv run ingest run --workers 4 2>&1 | tee ingest.log
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOGGER_NAME: Final = "ingester"


class _Formatter(logging.Formatter):
    """Include optional extra fields job_id, job_label for worker lines."""

    def format(self, record: logging.LogRecord) -> str:
        job_id = getattr(record, "job_id", None)
        job_label = getattr(record, "job_label", None)
        if job_id is not None and job_label:
            record.job_prefix = f"[job_id={job_id} {job_label}] "
        elif job_id is not None:
            record.job_prefix = f"[job_id={job_id}] "
        else:
            record.job_prefix = ""
        return super().format(record)


def configure_logging(level: str | None = None) -> None:
    """
    Attach a stderr handler to the 'ingester' logger tree (once per process).
    Always refreshes the log level from *level*, INGEST_LOG_LEVEL, or INFO.
    """
    root = logging.getLogger(_LOGGER_NAME)
    raw = (level or os.environ.get("INGEST_LOG_LEVEL", "INFO")).upper().strip()
    root.setLevel(getattr(logging, raw, logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            _Formatter(
                fmt=(
                    "%(asctime)s.%(msecs)03d %(levelname)-5s pid=%(process)d "
                    "%(job_prefix)s%(name)s: %(message)s"
                ),
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Child logger under 'ingester.*' (e.g. 'ingester.worker')."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def job_extra(job_id: int, symbol: str, interval: str, year: int, month: int) -> dict:
    """Pass as logger.info(..., extra=job_extra(...))."""
    label = f"{symbol}/{interval} {year:04d}-{month:02d}"
    return {"job_id": job_id, "job_label": label}
