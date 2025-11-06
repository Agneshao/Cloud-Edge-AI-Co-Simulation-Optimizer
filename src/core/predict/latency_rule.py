"""Rule-based latency prediction models."""

from typing import Dict, Any


def predict_latency(
    base_latency_ms: float,
    sku: str,
    precision: str,
    resolution: tuple[int, int],
    batch_size: int = 1
) -> float:
    """
    Predict latency based on SKU, precision, resolution, and batch size.
    
    Args:
        base_latency_ms: Base latency from profiling
        sku: Jetson SKU identifier (e.g., "orin_super", "orin_nano")
        precision: Model precision ("INT8", "FP16", "FP32")
        resolution: (height, width) tuple
        batch_size: Batch size
    
    Returns:
        Predicted latency in milliseconds
    """
    # SKU scaling factors (relative to Orin Super)
    sku_factors = {
        "orin_super": 1.0,
        "orin_nx": 0.8,
        "orin_nano": 0.6,
        "xavier_nx": 0.5,
        "nano": 0.3,
    }
    
    # Precision scaling factors
    precision_factors = {
        "INT8": 0.5,
        "FP16": 0.7,
        "FP32": 1.0,
    }
    
    # Resolution scaling (assume linear with pixel count)
    base_resolution = (640, 480)  # Reference resolution
    pixel_ratio = (resolution[0] * resolution[1]) / (base_resolution[0] * base_resolution[1])
    
    # Batch scaling (sub-linear)
    batch_factor = batch_size ** 0.8
    
    sku_factor = sku_factors.get(sku.lower(), 1.0)
    precision_factor = precision_factors.get(precision.upper(), 1.0)
    
    predicted_latency = (
        base_latency_ms
        * (1.0 / sku_factor)
        * precision_factor
        * pixel_ratio
        * batch_factor
    )
    
    return predicted_latency

