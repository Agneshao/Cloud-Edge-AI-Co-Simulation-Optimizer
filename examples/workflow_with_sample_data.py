"""
Complete workflow example using your sample profiling data format.

This script shows how to:
1. Parse your CSV data format
2. Convert to EdgeTwin format
3. Run predictions
4. Optimize configuration
5. Generate report

Your sample data format:
    timestamp, frame_id, engine_name, engine_precision, engine_batch, 
    engine_shape, end_to_end_ms, power_mW, gpu_temp_C, ram_usage_MB, ...
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.plan.reporter import ReportGenerator


def parse_sample_data_row(row: pd.Series) -> Dict[str, Any]:
    """
    Parse a single row from your sample data format.
    
    Example row:
        timestamp=1764086821, frame_id=11, engine_precision=FP16,
        engine_batch=1, engine_shape=[1,3,640,640], end_to_end_ms=31.492104,
        power_mW=5686, gpu_temp_C=47.937, ram_usage_MB=3423, ...
    """
    # Parse engine_shape (assuming format like "[1, 3, 640, 640]")
    shape_str = row.get('engine_shape', '[1, 3, 640, 640]')
    if isinstance(shape_str, str):
        # Remove brackets and split
        shape_str = shape_str.strip('[]')
        shape = [int(x.strip()) for x in shape_str.split(',')]
    else:
        shape = shape_str
    
    # Extract resolution (assuming NCHW: [batch, channels, height, width])
    if len(shape) >= 4:
        height, width = shape[2], shape[3]
    else:
        height, width = 640, 640
    
    # Determine SKU from jetson_mode or platform
    jetson_mode = str(row.get('jetson_mode', '')).lower()
    if '15w' in jetson_mode or '15W' in jetson_mode:
        sku = "orin_nx"  # Adjust based on your actual hardware
    else:
        sku = "orin_nx"  # Default, adjust as needed
    
    return {
        'timestamp': row.get('timestamp'),
        'frame_id': row.get('frame_id'),
        'model_name': row.get('engine_name', 'unknown'),
        'precision': row.get('engine_precision', 'FP16'),
        'batch_size': int(row.get('engine_batch', 1)),
        'resolution': (height, width),
        'latency_ms': float(row.get('end_to_end_ms', 0)),
        'trt_latency_ms': float(row.get('trt_latency_ms', 0)),
        'preprocess_ms': float(row.get('input_preprocess_ms', 0)),
        'postprocess_ms': float(row.get('postprocess_ms', 0)),
        'power_mW': float(row.get('power_mW', 0)),
        'power_w': float(row.get('power_mW', 0)) / 1000.0,
        'gpu_temp_C': float(row.get('gpu_temp_C', 0)),
        'ram_usage_MB': float(row.get('ram_usage_MB', 0)),
        'gpu_mem_alloc_MB': float(row.get('gpu_mem_alloc_MB', 0)),
        'gpu_util_percent': float(row.get('gpu_util_percent', 0)),
        'fps': 1000.0 / float(row.get('end_to_end_ms', 1)) if row.get('end_to_end_ms', 0) > 0 else 0,
        'sku': sku,
        'jetson_mode': row.get('jetson_mode', ''),
    }


def workflow_with_sample_data(csv_path: str):
    """
    Complete workflow using your sample data format.
    """
    print("=" * 80)
    print("EdgeTwin Workflow with Sample Data")
    print("=" * 80)
    
    # Step 1: Load and parse your data
    print("\n[Step 1] Loading sample data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"✗ File not found: {csv_path}")
        print("\nCreating example data row...")
        # Create example row matching your format
        example_row = {
            'timestamp': 1764086821,
            'frame_id': 11,
            'engine_name': 'yolov8n_fp16_static',
            'engine_precision': 'FP16',
            'engine_batch': 1,
            'engine_shape': '[1, 3, 640, 640]',
            'engine_size_MB': 8.999629974,
            'platform': 'Linux-5.15.148-tegra-aarch64-with-glibc2.35',
            'cuda_version': '12.2',
            'tensorrt_version': '10.3.0',
            'jetson_mode': 'NV Power Mode: 15W',
            'trt_latency_ms': 14.97836597,
            'input_preprocess_ms': 12.743995,
            'memcpy_h2d_ms': 1.640625007,
            'memcpy_d2h_ms': 2.053053002,
            'postprocess_ms': 0.0001919688657,
            'end_to_end_ms': 31.492104,
            'gpu_temp_C': 47.937,
            'gpu_freq_MHz': 305,
            'gpu_util_percent': 20,
            'power_mW': 5686,
            'ram_usage_MB': 3423,
            'swap_usage_MB': 0,
            'cpu_util_percent': 30.33333333,
            'gpu_mem_alloc_MB': 0,
            'sensor_time': 1764086821,
            'frame_drop': 0,
            'queue_delay_ms': 36.00406647,
            'frame_source_name': 'YOLOvideo',
        }
        df = pd.DataFrame([example_row])
        print("✓ Created example data row")
    
    # Parse first row as baseline
    baseline = parse_sample_data_row(df.iloc[0])
    
    print(f"\nBaseline Configuration:")
    print(f"  Model: {baseline['model_name']}")
    print(f"  Precision: {baseline['precision']}")
    print(f"  Resolution: {baseline['resolution']}")
    print(f"  Batch Size: {baseline['batch_size']}")
    print(f"  SKU: {baseline['sku']}")
    print(f"  Jetson Mode: {baseline['jetson_mode']}")
    
    print(f"\nBaseline Performance:")
    print(f"  Latency: {baseline['latency_ms']:.2f} ms")
    print(f"  FPS: {baseline['fps']:.1f}")
    print(f"  Power: {baseline['power_w']:.2f} W ({baseline['power_mW']:.0f} mW)")
    print(f"  GPU Temp: {baseline['gpu_temp_C']:.1f} °C")
    print(f"  RAM Usage: {baseline['ram_usage_MB']:.0f} MB")
    
    # Step 2: Build profile results
    print("\n[Step 2] Building profile results...")
    profile_results = {
        "latency_ms": {
            "total": baseline['latency_ms'],
            "preprocess": baseline['preprocess_ms'],
            "inference": baseline['trt_latency_ms'],
            "postprocess": baseline['postprocess_ms'],
        },
        "latency_stats": {
            "total": {
                "mean": baseline['latency_ms'],
                "std": 0.0,  # Single measurement
                "min": baseline['latency_ms'],
                "max": baseline['latency_ms'],
            }
        },
        "fps": baseline['fps'],
        "power_w": baseline['power_w'],
        "memory_mb": baseline['ram_usage_MB'],
        "sku": baseline['sku'],
        "iterations": 1,
    }
    print("✓ Profile results built")
    
    # Step 3: Test predictions for different configurations
    print("\n[Step 3] Testing predictions for different configurations...")
    base_latency_ms = baseline['latency_ms']
    base_power_w = baseline['power_w']
    sku = baseline['sku']
    power_mode = "15W" if "15W" in baseline['jetson_mode'] else "MAXN"
    
    test_configs = [
        {"precision": "INT8", "resolution": (640, 640), "batch_size": 1, "name": "INT8 @ 640x640"},
        {"precision": "FP16", "resolution": (640, 640), "batch_size": 1, "name": "FP16 @ 640x640 (baseline)"},
        {"precision": "FP16", "resolution": (1280, 1280), "batch_size": 1, "name": "FP16 @ 1280x1280"},
        {"precision": "FP16", "resolution": (640, 640), "batch_size": 2, "name": "FP16 @ 640x640 (batch=2)"},
    ]
    
    print(f"\n{'Configuration':<40} {'Latency (ms)':<15} {'Power (W)':<15} {'FPS':<10}")
    print("-" * 80)
    
    predictions = {}
    for config in test_configs:
        pred_latency = predict_latency(
            base_latency_ms=base_latency_ms,
            sku=sku,
            precision=config["precision"],
            resolution=config["resolution"],
            batch_size=config["batch_size"]
        )
        
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 0
        pred_power = predict_power(
            base_power_w=base_power_w,
            fps=pred_fps,
            precision=config["precision"],
            sku=sku,
            resolution=config["resolution"],
            power_mode=power_mode
        )
        
        print(f"{config['name']:<40} {pred_latency:<15.2f} {pred_power:<15.2f} {pred_fps:<10.1f}")
        
        # Store baseline prediction
        if config['name'] == "FP16 @ 640x640 (baseline)":
            predictions = {
                "latency_ms": pred_latency,
                "power_w": pred_power,
            }
    
    # Step 4: Optimization
    print("\n[Step 4] Running optimization...")
    
    def objective(knobs: ConfigKnobs) -> float:
        """Objective: Minimize latency while keeping power < 15W."""
        pred_latency = predict_latency(
            base_latency_ms, sku, knobs.precision,
            knobs.resolution, knobs.batch_size
        )
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
        pred_power = predict_power(
            base_power_w, pred_fps, knobs.precision,
            sku, knobs.resolution, power_mode=power_mode
        )
        
        # Hard constraint: Power must be < 15W
        if pred_power > 15.0:
            return 10000.0
        
        # Minimize latency (with small power penalty)
        return pred_latency + 0.1 * pred_power
    
    best_knobs = greedy_search(
        objective_fn=objective,
        initial_knobs=ConfigKnobs(
            precision=baseline['precision'],
            resolution=baseline['resolution'],
            batch_size=baseline['batch_size']
        ),
        max_iterations=20
    )
    
    # Evaluate best configuration
    best_latency = predict_latency(
        base_latency_ms, sku, best_knobs.precision,
        best_knobs.resolution, best_knobs.batch_size
    )
    best_fps = 1000.0 / best_latency if best_latency > 0 else 0
    best_power = predict_power(
        base_power_w, best_fps, best_knobs.precision,
        sku, best_knobs.resolution, power_mode=power_mode
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"\nBest Configuration:")
    print(f"  Precision: {best_knobs.precision}")
    print(f"  Resolution: {best_knobs.resolution}")
    print(f"  Batch Size: {best_knobs.batch_size}")
    print(f"  Frame Skip: {best_knobs.frame_skip}")
    print(f"\nPredicted Performance:")
    print(f"  Latency: {best_latency:.2f} ms")
    print(f"  FPS: {best_fps:.1f}")
    print(f"  Power: {best_power:.2f} W")
    print(f"  Objective Score: {objective(best_knobs):.2f}")
    
    # Step 5: Thermal prediction
    print("\n[Step 5] Thermal prediction...")
    thermal_model = ThermalRC(
        ambient_temp_c=25.0,
        thermal_resistance_c_per_w=0.5,
        thermal_capacitance_j_per_c=10.0,
        max_temp_c=70.0
    )
    time_to_throttle = thermal_model.time_to_throttle(best_power)
    if time_to_throttle != float('inf'):
        print(f"  Time to throttle: {time_to_throttle:.1f} s")
    else:
        print(f"  Time to throttle: Never (power is safe)")
    
    # Step 6: Generate report
    print("\n[Step 6] Generating report...")
    report_gen = ReportGenerator()
    
    # Update predictions with best config
    predictions = {
        "latency_ms": best_latency,
        "power_w": best_power,
        "time_to_throttle_s": time_to_throttle if time_to_throttle != float('inf') else None,
    }
    
    optimized_config = best_knobs.to_dict()
    
    report_path = report_gen.generate_report(
        profile_results=profile_results,
        predictions=predictions,
        optimized_config=optimized_config
    )
    
    print(f"✓ Report generated: {report_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Workflow Summary")
    print("=" * 80)
    print(f"Baseline: {baseline['precision']} @ {baseline['resolution']} → {baseline['latency_ms']:.2f}ms @ {baseline['power_w']:.2f}W")
    print(f"Optimized: {best_knobs.precision} @ {best_knobs.resolution} → {best_latency:.2f}ms @ {best_power:.2f}W")
    improvement = ((baseline['latency_ms'] - best_latency) / baseline['latency_ms']) * 100
    print(f"Improvement: {improvement:+.1f}% latency reduction")
    print(f"Report: {report_path}")
    print("=" * 80)
    
    return {
        'baseline': baseline,
        'best_knobs': best_knobs,
        'best_latency': best_latency,
        'best_power': best_power,
        'report_path': report_path,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EdgeTwin workflow with sample data")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to your profiling CSV file (if not provided, uses example data)"
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv if args.csv else "data/jetbenchdb/profiles_local.csv"
    
    try:
        results = workflow_with_sample_data(csv_path)
        print("\n✓ Workflow completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

