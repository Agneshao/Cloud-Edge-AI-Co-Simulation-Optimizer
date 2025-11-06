"""Tools for validating and comparing power models with empirical data.

TODO: Validation workflow:
  1. Collect power measurements from Jetson hardware (use tegrastats)
  2. Save measurements to CSV (see power_measurements_template.csv)
  3. Load measurements using load_measurements_from_csv()
  4. Compare models using compare_models() to see which is most accurate
  5. Calibrate coefficients using calibrate_model_coefficients()
  6. Update power_models.py with calibrated coefficients

TODO: Understanding metrics:
  - MAE (Mean Absolute Error): Average prediction error (watts)
  - RMSE (Root Mean Squared Error): Penalizes large errors more
  - R² (R-squared): How well model fits data (0-1, higher is better)
  - Mean Relative Error: Percentage error (useful for comparing across scales)
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from .power_models import PowerModelType, predict_power


@dataclass
class PowerMeasurement:
    """Single power measurement with metadata."""
    power_w: float
    fps: float
    precision: str
    sku: str
    resolution: tuple[int, int]
    power_mode: str = "MAXN"
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class ModelMetrics:
    """Metrics for evaluating model accuracy."""
    mae: float  # Mean Absolute Error (watts)
    rmse: float  # Root Mean Squared Error (watts)
    r2: float  # R-squared (0-1, higher is better)
    mean_relative_error: float  # Mean relative error (%)
    max_error: float  # Maximum absolute error (watts)


def calculate_metrics(
    predictions: List[float],
    actuals: List[float]
) -> ModelMetrics:
    """
    Calculate evaluation metrics for power predictions.
    
    TODO: Understanding these metrics:
      - MAE: Easy to interpret ("average error is X watts")
      - RMSE: More sensitive to outliers (larger errors penalized more)
      - R²: 1.0 = perfect fit, 0.0 = no better than mean, <0 = worse than mean
      - Relative error: Useful when power varies widely
    
    Args:
        predictions: List of predicted power values
        actuals: List of actual measured power values
    
    Returns:
        ModelMetrics object
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - actuals))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Mean relative error
    relative_errors = np.abs((predictions - actuals) / (actuals + 1e-6)) * 100
    mean_relative_error = np.mean(relative_errors)
    
    # Maximum error
    max_error = np.max(np.abs(predictions - actuals))
    
    return ModelMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mean_relative_error=mean_relative_error,
        max_error=max_error
    )


def compare_models(
    measurements: List[PowerMeasurement],
    base_power_w: float,
    model_types: Optional[List[PowerModelType]] = None
) -> Dict[PowerModelType, ModelMetrics]:
    """
    Compare multiple power models against empirical data.
    
    TODO: How to use:
      1. Collect at least 10-20 measurements (more is better)
      2. Call this function with your measurements
      3. Look at R² to see which model fits best
      4. Look at RMSE to see which has lowest error
      5. Use the best model for predictions
    
    Args:
        measurements: List of power measurements
        base_power_w: Base power for predictions
        model_types: List of models to compare (defaults to all)
    
    Returns:
        Dictionary mapping model type to metrics
    """
    if model_types is None:
        model_types = list(PowerModelType)
    
    results = {}
    
    for model_type in model_types:
        predictions = []
        actuals = []
        
        for meas in measurements:
            pred = predict_power(
                base_power_w=base_power_w,
                fps=meas.fps,
                precision=meas.precision,
                sku=meas.sku,
                resolution=meas.resolution,
                model_type=model_type,
                power_mode=meas.power_mode
            )
            predictions.append(pred)
            actuals.append(meas.power_w)
        
        metrics = calculate_metrics(predictions, actuals)
        results[model_type] = metrics
    
    return results


def load_measurements_from_csv(csv_path: str) -> List[PowerMeasurement]:
    """
    Load power measurements from CSV file.
    
    TODO: Check our CSV format:
      power_w,fps,precision,sku,resolution_h,resolution_w,power_mode,gpu_util,cpu_util
      
    Use power_measurements_template.csv as a reference.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of PowerMeasurement objects
    """
    import csv
    
    measurements = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip comment lines
            if row.get('power_w', '').startswith('#'):
                continue
            
            measurements.append(PowerMeasurement(
                power_w=float(row['power_w']),
                fps=float(row['fps']),
                precision=row['precision'],
                sku=row['sku'],
                resolution=(int(row['resolution_h']), int(row['resolution_w'])),
                power_mode=row.get('power_mode', 'MAXN'),
                gpu_utilization=float(row['gpu_util']) if row.get('gpu_util') else None,
                cpu_utilization=float(row['cpu_util']) if row.get('cpu_util') else None,
            ))
    
    return measurements
