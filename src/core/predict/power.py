"""Power consumption prediction models.

Simple interface for power prediction. The models use estimated coefficients
until you collect empirical data to calibrate them.

TODO: Power modeling basics:
  1. Start simple: Use SIMPLE_LINEAR model first
  2. Collect data: Measure power on real Jetson hardware
  3. Validate: Compare predictions to measurements
  4. Improve: Calibrate coefficients or switch models
  5. Iterate: More data = better models

See power_models.py for model implementations.
See power_validation.py for validation tools.
"""

from typing import Dict, Any
from .power_models import predict_power as _predict_power, PowerModelType


def predict_power(
    base_power_w: float,
    fps: float,
    precision: str,
    sku: str,
    resolution: tuple[int, int],
    model_type: PowerModelType = PowerModelType.POWER_MODE_AWARE,
    **kwargs
) -> float:
    """
    Predict power consumption based on workload characteristics.
    
    TODO: How to use this:
      1. Start with default POWER_MODE_AWARE model
      2. Provide base_power_w from idle measurements
      3. Adjust power_mode if running in constrained mode
      4. Compare predictions to actual measurements
      5. Use power_validation tools to improve accuracy
    
    Args:
        base_power_w: Base power consumption (idle + static)
        fps: Frames per second
        precision: Model precision ("INT8", "FP16", "FP32")
        sku: Jetson SKU identifier
        resolution: (height, width) tuple
        model_type: Power model to use (default: POWER_MODE_AWARE)
        **kwargs: Additional arguments (e.g., power_mode="15W")
    
    Returns:
        Predicted power in watts
    
    Example:
        >>> # Simple usage (recommended)
        >>> power = predict_power(10.0, 30.0, "FP16", "orin_super", (640, 480))
        
        >>> # With power mode
        >>> power = predict_power(10.0, 30.0, "FP16", "orin_super", (640, 480), 
        ...                      power_mode="15W")
        
        >>> # Use simple linear model (for learning/experimentation)
        >>> from src.core.predict.power_models import PowerModelType
        >>> power = predict_power(10.0, 30.0, "FP16", "orin_super", (640, 480),
        ...                      model_type=PowerModelType.SIMPLE_LINEAR)
    """
    return _predict_power(base_power_w, fps, precision, sku, resolution, model_type, **kwargs)
