"""Co-simulation module for EdgeTwin.

This module provides the core co-simulation loop that integrates hardware latency
predictions with environment simulation to demonstrate how hardware behavior affects
task performance.
"""

from .env_base import AbstractEnv
from .env_toy import ToyEnv
from .loop import run_cosim
from .metrics import evaluate_logs
from .workload import dummy_workload

__all__ = [
    "AbstractEnv",
    "ToyEnv",
    "run_cosim",
    "evaluate_logs",
    "dummy_workload",
]

