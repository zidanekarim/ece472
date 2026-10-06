"""Evaluation entry point for hw04."""

import structlog

from .logging import configure_logging

from pathlib import Path
from flax import nnx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from .config import TrainingSettings
from .data import Data
from .model import Classifier
from .training import create_learning_rate_schedule, train

def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    log = structlog.get_logger()
    log.info(
        "Evaluation not yet implemented for hw04. Implement evaluation logic in src/hw04/evaluate.py"
    )
