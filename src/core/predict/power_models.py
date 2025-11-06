"""Simple power consumption prediction models.

NOTE: These models use estimated coefficients. To get accurate predictions,
collect empirical data from real Jetson hardware and calibrate the coefficients.
See power_validation.py for tools to help with calibration.

TODO: Once we have empirical power measurements:
  1. Collect data using tegrastats on Jetson hardware
  2. Use power_validation.calibrate_model_coefficients() to fit coefficients
  3. Replace the hardcoded coefficients below with calibrated values
  4. Compare models using power_validation.compare_models()
"""

from typing import Dict, Any, Optional
from enum import Enum


class PowerModelType(Enum):
    """Types of power models available."""
    SIMPLE_LINEAR = "simple_linear"  # Current simple model
    # UTILIZATION_BASED = "utilization_based"  # Based on GPU/CPU utilization
    # COMPONENT_BASED = "component_based"  # P = P_static + P_CPU + P_GPU + P_MEM
    POWER_MODE_AWARE = "power_mode_aware"  # Accounts for power mode settings
    # POLYNOMIAL = "polynomial"  # Non-linear polynomial model


def predict_power_simple_linear(
    base_power_w: float,
    fps: float,
    precision: str,
    sku: str,
    resolution: tuple[int, int]
) -> float:
    """
    Simple linear model: base_power + k * fps * workload_factor
    
    This is the simplest model. Good starting point, but may not capture
    non-linear power behavior at high utilization.
    
    TODO: Try fitting this model to the data:
      - What value of k best fits our measurements?
      - Does the linear assumption hold across all FPS ranges?
      - Consider adding non-linear terms if errors are high at high FPS
    
    Args:
        base_power_w: Base power consumption (idle + static)
        fps: Frames per second
        precision: Model precision ("INT8", "FP16", "FP32")
        sku: Jetson SKU identifier
        resolution: (height, width) tuple
    
    Returns:
        Predicted power in watts
    """
    # Precision power factors (estimated - needs calibration)
    precision_factors = {
        "INT8": 0.6,  # INT8 typically uses ~60% of FP32 power
        "FP16": 0.8,  # FP16 typically uses ~80% of FP32 power
        "FP32": 1.0,  # FP32 is baseline
    }
    
    # Resolution factor (normalized to 640x480)
    base_resolution = (640, 480)
    pixel_ratio = (resolution[0] * resolution[1]) / (base_resolution[0] * base_resolution[1])
    
    # Workload factor combines precision and resolution
    precision_factor = precision_factors.get(precision.upper(), 1.0)
    workload_factor = precision_factor * (pixel_ratio ** 0.7)  # Sub-linear with resolution
    
    # Power scaling coefficient (watts per fps per workload unit)
    # TODO: Calibrate this coefficient from empirical data!
    #   - Measure power at different FPS values
    #   - Use linear regression to find best k
    k = 0.1  # Estimated coefficient - replace with calibrated value
    
    predicted_power = base_power_w + (k * fps * workload_factor)
    
    return predicted_power


def predict_power_power_mode_aware(
    base_power_w: float,
    fps: float,
    precision: str,
    sku: str,
    resolution: tuple[int, int],
    power_mode: str = "MAXN"
) -> float:
    """
    Power mode aware model with non-linear scaling.
    
    Accounts for different power limits in different power modes (MAXN, 15W, etc.)
    and uses non-linear scaling to better capture power behavior.
    
    TODO: Enhance this model:
      1. Add GPU/CPU utilization as inputs (from tegrastats)
      2. Model power saturation at high utilization
      3. Add thermal throttling effects
      4. Calibrate alpha and k values from data
    
    Args:
        base_power_w: Base power consumption
        fps: Frames per second
        precision: Model precision
        sku: Jetson SKU identifier
        resolution: (height, width) tuple
        power_mode: Power mode ("MAXN", "15W", "10W", "5W")
    
    Returns:
        Predicted power in watts
    """
    # Power mode limits (from jetson_devices.yaml)
    # TODO: Verify these limits match your hardware
    power_limits = {
        "orin_super": {"MAXN": 100.0, "15W": 15.0},
        "orin_nx": {"MAXN": 25.0, "15W": 15.0},
        "orin_nano": {"MAXN": 15.0, "10W": 10.0},
        "xavier_nx": {"MAXN": 25.0, "15W": 15.0},
        "nano": {"MAXN": 10.0, "5W": 5.0},
    }
    
    # Power mode specific coefficients
    # TODO: Calibrate these from empirical data!
    #   - Measure power at different power modes
    #   - Fit k and alpha for each mode
    mode_coefficients = {
        "MAXN": {"k": 0.12, "alpha": 1.4},  # Higher scaling, moderate curve
        "15W": {"k": 0.08, "alpha": 1.6},    # Lower scaling, steeper curve
        "10W": {"k": 0.06, "alpha": 1.7},
        "5W": {"k": 0.04, "alpha": 1.8},
    }
    
    coeffs = mode_coefficients.get(power_mode, mode_coefficients["MAXN"])
    p_max = power_limits.get(sku.lower(), {}).get(power_mode, 25.0)
    
    # Workload factor
    precision_factors = {"INT8": 0.6, "FP16": 0.8, "FP32": 1.0}
    precision_factor = precision_factors.get(precision.upper(), 1.0)
    
    base_resolution = (640, 480)
    pixel_ratio = (resolution[0] * resolution[1]) / (base_resolution[0] * base_resolution[1])
    workload_factor = precision_factor * (pixel_ratio ** 0.7)
    
    # Non-linear model: P = P_base + k * (fps * workload_factor)^alpha
    # TODO: Why non-linear? Research power consumption curves:
    #   - Power doesn't scale linearly with workload
    #   - Efficiency decreases at high utilization
    #   - Alpha > 1 captures this effect
    workload_intensity = fps * workload_factor
    predicted_power = base_power_w + coeffs["k"] * (workload_intensity ** coeffs["alpha"])
    
    # Cap at power limit
    predicted_power = min(predicted_power, p_max)
    
    return predicted_power


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
    Unified power prediction interface.
    
    TODO: Experiment with different models:
      1. Start with SIMPLE_LINEAR to understand basics
      2. Try POWER_MODE_AWARE for better accuracy
      3. Once you have data, compare models using power_validation.compare_models()
      4. Choose the model with best R² and lowest RMSE
    
    Args:
        base_power_w: Base/idle power consumption
        fps: Frames per second
        precision: Model precision ("INT8", "FP16", "FP32")
        sku: Jetson SKU identifier
        resolution: (height, width) tuple
        model_type: Which power model to use (default: POWER_MODE_AWARE)
        **kwargs: Additional arguments (e.g., power_mode)
    
    Returns:
        Predicted power in watts
    """
    if model_type == PowerModelType.SIMPLE_LINEAR:
        return predict_power_simple_linear(base_power_w, fps, precision, sku, resolution)
    
    elif model_type == PowerModelType.POWER_MODE_AWARE:
        power_mode = kwargs.get("power_mode", "MAXN")
        return predict_power_power_mode_aware(
            base_power_w, fps, precision, sku, resolution, power_mode
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
