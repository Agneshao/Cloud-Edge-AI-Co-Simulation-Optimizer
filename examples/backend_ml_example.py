"""
Example script demonstrating backend and ML components.

This script shows how to:
1. Use prediction models (latency, power, thermal)
2. Run optimization algorithms
3. Build features for ML models
4. Test the API endpoints

Run this script to understand how the components work together.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.predict.features import build_features
from src.core.optimize.knobs import ConfigKnobs, get_knob_bounds
from src.core.optimize.search import greedy_search


def example_predictions():
    """Example: Using prediction models."""
    print("=" * 60)
    print("Example 1: Prediction Models")
    print("=" * 60)
    
    # Base measurements (from profiling)
    base_latency_ms = 50.0
    base_power_w = 10.0
    sku = "orin_super"
    
    # Configuration to predict
    precision = "FP16"
    resolution = (640, 480)
    batch_size = 1
    fps = 30.0
    
    # Predict latency
    predicted_latency = predict_latency(
        base_latency_ms=base_latency_ms,
        sku=sku,
        precision=precision,
        resolution=resolution,
        batch_size=batch_size
    )
    print(f"Predicted Latency: {predicted_latency:.2f} ms")
    
    # Predict power
    predicted_power = predict_power(
        base_power_w=base_power_w,
        fps=fps,
        precision=precision,
        sku=sku,
        resolution=resolution,
        power_mode="MAXN"
    )
    print(f"Predicted Power: {predicted_power:.2f} W")
    
    # Predict thermal (time to throttle)
    thermal_model = ThermalRC(
        ambient_temp_c=25.0,
        thermal_resistance_c_per_w=0.5,
        thermal_capacitance_j_per_c=10.0,
        max_temp_c=70.0
    )
    time_to_throttle = thermal_model.time_to_throttle(predicted_power)
    print(f"Time to Throttle: {time_to_throttle:.1f} s" if time_to_throttle != float('inf') else "Time to Throttle: Never")
    
    print()


def example_optimization():
    """Example: Using optimization algorithms."""
    print("=" * 60)
    print("Example 2: Optimization")
    print("=" * 60)
    
    # Define objective function
    # This is a simple example - minimize weighted combination of latency and power
    def objective(knobs: ConfigKnobs) -> float:
        """Objective: minimize latency + 0.1 * power."""
        # Mock predictions (in real code, use actual prediction functions)
        latency = 50.0 * (1.0 if knobs.precision == "FP16" else 0.5 if knobs.precision == "INT8" else 1.5)
        power = 10.0 + (knobs.resolution[0] * knobs.resolution[1] / (640 * 480)) * 5.0
        
        return latency + 0.1 * power
    
    # Run greedy search
    print("Running greedy search optimization...")
    best_knobs = greedy_search(
        objective_fn=objective,
        initial_knobs=ConfigKnobs(precision="FP16", resolution=(640, 480), batch_size=1),
        max_iterations=10
    )
    
    print(f"Best Configuration:")
    print(f"  Precision: {best_knobs.precision}")
    print(f"  Resolution: {best_knobs.resolution}")
    print(f"  Batch Size: {best_knobs.batch_size}")
    print(f"  Frame Skip: {best_knobs.frame_skip}")
    print(f"  Objective Value: {objective(best_knobs):.2f}")
    print()


def example_feature_engineering():
    """Example: Building features for ML models."""
    print("=" * 60)
    print("Example 3: Feature Engineering")
    print("=" * 60)
    
    # Profile data (from actual profiling)
    profile_data = {
        "latency_ms": 50.0,
        "power_w": 15.0,
        "memory_mb": 512.0
    }
    
    # Configuration knobs
    knobs = {
        "precision": "FP16",
        "resolution": (640, 480),
        "batch_size": 1,
        "frame_skip": 0
    }
    
    # Build feature vector
    features = build_features(profile_data, knobs)
    print(f"Feature Vector Shape: {features.shape}")
    print(f"Features: {features}")
    print(f"Feature Names: [latency_ms, power_w, memory_mb, precision, resolution_height, resolution_width, batch_size, frame_skip]")
    print()


def example_comparison():
    """Example: Comparing different configurations."""
    print("=" * 60)
    print("Example 4: Configuration Comparison")
    print("=" * 60)
    
    base_latency = 50.0
    base_power = 10.0
    sku = "orin_super"
    fps = 30.0
    
    configs = [
        {"precision": "INT8", "resolution": (640, 480), "name": "INT8 @ 640x480"},
        {"precision": "FP16", "resolution": (640, 480), "name": "FP16 @ 640x480"},
        {"precision": "FP16", "resolution": (1280, 960), "name": "FP16 @ 1280x960"},
    ]
    
    print(f"{'Configuration':<25} {'Latency (ms)':<15} {'Power (W)':<15} {'FPS':<10}")
    print("-" * 65)
    
    for config in configs:
        latency = predict_latency(
            base_latency, sku, config["precision"], config["resolution"]
        )
        power = predict_power(
            base_power, fps, config["precision"], sku, config["resolution"]
        )
        estimated_fps = 1000.0 / latency if latency > 0 else 0
        
        print(f"{config['name']:<25} {latency:<15.2f} {power:<15.2f} {estimated_fps:<10.1f}")
    
    print()


def example_knob_bounds():
    """Example: Understanding knob bounds."""
    print("=" * 60)
    print("Example 5: Configuration Knob Bounds")
    print("=" * 60)
    
    bounds = get_knob_bounds()
    print("Available Configuration Options:")
    print(f"  Precision: {bounds['precision']}")
    print(f"  Resolution Height: {bounds['resolution_height']}")
    print(f"  Resolution Width: {bounds['resolution_width']}")
    print(f"  Batch Size: {bounds['batch_size']}")
    print(f"  Frame Skip: {bounds['frame_skip']}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("EdgeTwin Backend & ML Components - Examples")
    print("=" * 60 + "\n")
    
    try:
        example_predictions()
        example_optimization()
        example_feature_engineering()
        example_comparison()
        example_knob_bounds()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Explore the source code in src/core/predict/ and src/core/optimize/")
        print("2. Try modifying the objective function in example_optimization()")
        print("3. Add more features to build_features() in src/core/predict/features.py")
        print("4. Wire up the API endpoints in src/apps/api/server.py")
        print("5. Check out docs/GETTING_STARTED_BACKEND_ML.md for more guidance")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

