"""Dummy workload to simulate YOLO compute operations.

This module provides compute-intensive operations that simulate the actual
workload of running a perception model (e.g., YOLO) on edge hardware.
"""

import numpy as np


def dummy_workload(size: int = 2000) -> None:
    """Simulate compute load similar to YOLO inference.
    
    This function performs matrix multiplication to consume CPU/GPU resources,
    creating realistic compute load that would occur during actual model inference.
    
    Args:
        size: Size of the square matrix to multiply (default: 2000)
    """
    # Simulate compute load with matrix multiplication
    x = np.random.randn(size, size)
    _ = x @ x

