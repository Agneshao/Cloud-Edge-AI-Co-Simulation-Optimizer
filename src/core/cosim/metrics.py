"""Metrics and evaluation for co-simulation results."""

from typing import Tuple
import pandas as pd


def evaluate_logs(logs: list[dict]) -> Tuple[pd.DataFrame, bool]:
    """Evaluate co-simulation logs and compute metrics.
    
    Args:
        logs: List of log dictionaries from run_cosim()
        
    Returns:
        Tuple of (DataFrame with logs, collision flag)
        - DataFrame contains all logged metrics
        - collision is True if |offset| > 1.0 at any point
    """
    df = pd.DataFrame(logs)
    
    # Check if collision occurred (offset exceeded threshold)
    collided = (df["offset"].abs() > 1.0).any()
    
    return df, collided

