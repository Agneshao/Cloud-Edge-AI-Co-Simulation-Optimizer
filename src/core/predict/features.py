"""Build features from profile data and configuration knobs."""

from typing import Dict, Any
import numpy as np


def build_features(profile_data: Dict[str, Any], knobs: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector from profile data and configuration knobs.
    
    Args:
        profile_data: Profile results (latency, power, etc.)
        knobs: Configuration knobs (precision, resolution, etc.)
    
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # Extract profile features
    features.append(profile_data.get("latency_ms", 0.0))
    features.append(profile_data.get("power_w", 0.0))
    features.append(profile_data.get("memory_mb", 0.0))
    
    # Extract knob features
    features.append(knobs.get("precision", 0))  # 0=INT8, 1=FP16, 2=FP32
    features.append(knobs.get("resolution", 0))  # e.g., height or width
    features.append(knobs.get("batch_size", 1))
    features.append(knobs.get("frame_skip", 0))
    
    return np.array(features, dtype=np.float32)

