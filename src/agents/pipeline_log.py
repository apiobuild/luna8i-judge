from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import select

from src.db import get_session
from src.schemas.db import JobConfig


def _log(
    job_id: str,
    node: str,
    message: str,
    level: str = "info",
    logger: logging.Logger | None = None,
    log_only: bool = False,
) -> None:
    """Write a log entry to the job's pipeline_log in SQLite and/or emit via logger.

    log_only=True emits via logger without touching the DB (useful for high-frequency
    per-row progress that doesn't need to be persisted).
    """
    if logger is not None:
        logger.log(getattr(logging, level.upper(), logging.INFO), "Job %s [%s]: %s", job_id, node, message)

    if log_only:
        return

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "node": node,
        "message": message,
    }
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            return
        log = list(job.pipeline_log or [])
        log.append(entry)
        job.pipeline_log = log
        job.updated_at = datetime.now(timezone.utc).isoformat()
        session.add(job)
        session.commit()
