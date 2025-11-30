"""Build features from profile data and configuration knobs."""

from typing import Dict, Any
import numpy as np


def build_features(profile_data: Dict[str, Any], knobs: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector from profile data and configuration knobs.
    
    Args:
        profile_data: Profile results (latency, power, etc.)
            - latency_ms can be a float or dict with "total" key
            - power_w: float
            - memory_mb: float
        knobs: Configuration knobs (precision, resolution, etc.)
    
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # Extract profile features
    # Handle latency_ms as either scalar or nested dict
    latency_ms = profile_data.get("latency_ms", 0.0)
    if isinstance(latency_ms, dict):
        latency_ms = latency_ms.get("total", 0.0)
    features.append(float(latency_ms))
    
    features.append(float(profile_data.get("power_w", 0.0)))
    features.append(float(profile_data.get("memory_mb", 0.0)))
    
    # Extract knob features
    # Convert precision to numeric: INT8=0, FP16=1, FP32=2
    precision_map = {"INT8": 0, "FP16": 1, "FP32": 2}
    precision_str = knobs.get("precision", "FP16")
    features.append(float(precision_map.get(precision_str.upper(), 1)))
    
    # Extract resolution (flatten tuple to height and width)
    resolution = knobs.get("resolution", (640, 480))
    if isinstance(resolution, (tuple, list)) and len(resolution) >= 2:
        features.append(float(resolution[0]))  # height
        features.append(float(resolution[1]))  # width
    else:
        features.append(640.0)
        features.append(480.0)
    
    features.append(float(knobs.get("batch_size", 1)))
    features.append(float(knobs.get("frame_skip", 0)))
    
    return np.array(features, dtype=np.float32)

