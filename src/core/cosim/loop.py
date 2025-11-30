"""Co-simulation main loop.

This module implements the core co-simulation loop that integrates:
1. Perception compute (dummy workload)
2. Hardware latency prediction
3. Environment stepping with delayed actions
"""

import time
from typing import Dict, Any, List
from .env_base import AbstractEnv
from .workload import dummy_workload
from ..predict.latency_rule import predict_latency


def run_cosim(
    env: AbstractEnv,
    steps: int = 300,
    base_latency_ms: float = 50.0,
    sku: str = "orin_nano",
    precision: str = "FP16",
    resolution: tuple[int, int] = (640, 480),
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Run co-simulation loop with hardware latency injection.
    
    The loop:
    1. Runs dummy workload (simulates YOLO compute)
    2. Predicts hardware latency based on configuration
    3. Sleeps for predicted latency (simulates hardware delay)
    4. Steps environment with delayed action
    5. Logs metrics for analysis
    
    Args:
        env: Environment instance (must implement AbstractEnv)
        steps: Maximum number of simulation steps
        base_latency_ms: Base latency from profiling (ms)
        sku: Jetson SKU identifier (e.g., "orin_super", "orin_nano")
        precision: Model precision ("INT8", "FP16", "FP32")
        resolution: Input resolution (height, width)
        batch_size: Batch size for inference
        
    Returns:
        List of log dictionaries containing step metrics
    """
    obs = env.reset()
    logs: List[Dict[str, Any]] = []
    
    for step in range(steps):
        # Simulate perception compute (YOLO inference)
        dummy_workload()
        
        # Simple P controller: action proportional to negative offset
        action = -obs["offset"]
        
        # Predict hardware latency based on configuration
        latency_ms = predict_latency(
            base_latency_ms=base_latency_ms,
            sku=sku,
            precision=precision,
            resolution=resolution,
            batch_size=batch_size,
        )
        
        # Inject hardware delay (simulate actual hardware latency)
        time.sleep(latency_ms / 1000.0)
        
        # Step environment with delayed action
        dt = max(latency_ms / 1000.0, 0.01)  # Ensure minimum dt
        obs, done = env.step(action, dt)
        
        # Log metrics
        logs.append({
            "step": step,
            "offset": obs["offset"],
            "dt": dt,
            "latency_ms": latency_ms,
            "t": obs["t"],
        })
        
        if done:
            break
    
    return logs

