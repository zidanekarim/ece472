"""Evaluation entry point for hw05."""

import structlog

from .logging import configure_logging


def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    log = structlog.get_logger()
    log.info(
        "Evaluation not yet implemented for hw05. Implement evaluation logic in src/hw05/evaluate.py"
    )
