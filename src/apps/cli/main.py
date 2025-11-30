"""CLI that wires profile→predict→optimize→report."""

import argparse
from pathlib import Path
import sys
import json
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.profile.pipeline_profiler import PipelineProfiler
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.optimize.model_converter import optimize_model_for_metrics
from src.core.plan.reporter import ReportGenerator
from src.core.utils.data_utils import load_profile_results, convert_sample_data_to_edgetwin


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EdgeTwin: Hardware-aware co-simulation for robotics AI"
    )
    
    parser.add_argument(
        "command",
        choices=["profile", "predict", "optimize", "optimize-model", "full", "convert-data"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to ONNX model"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video"
    )
    
    parser.add_argument(
        "--sku",
        type=str,
        default="orin_super",
        help="Jetson SKU identifier"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for reports (or output file for convert-data)"
    )
    
    # Add convert-data specific args
    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (for convert-data command)"
    )
    
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Target FPS (for optimize-model command)"
    )
    
    parser.add_argument(
        "--target-latency-ms",
        type=float,
        help="Target latency in milliseconds (for optimize-model command)"
    )
    
    parser.add_argument(
        "--max-power-w",
        type=float,
        help="Maximum power in watts (for optimize-model command)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "full":
            return run_full_workflow(args)
        elif args.command == "profile":
            return run_profile(args)
        elif args.command == "predict":
            return run_predict(args)
        elif args.command == "optimize":
            return run_optimize(args)
        elif args.command == "optimize-model":
            return run_optimize_model(args)
        elif args.command == "convert-data":
            from src.apps.cli.data_commands import convert_data
            return convert_data(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_full_workflow(args):
    """Run complete workflow: profile → predict → optimize → report."""
    print("=" * 60)
    print("EdgeTwin Full Workflow")
    print("=" * 60)
    
    if not args.model:
        print("Error: --model is required for full workflow", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output) if args.output else Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Profile
    print("\n[1/4] Profiling model...")
    profiler = PipelineProfiler(sku=args.sku)
    profile_results = profiler.profile(
        model_path=args.model,
        video_path=args.video,
        iterations=10
    )
    print(f"  ✓ Latency: {profile_results['latency_ms']['total']:.2f} ms")
    print(f"  ✓ FPS: {profile_results['fps']:.1f}")
    print(f"  ✓ Power: {profile_results['power_w']:.2f} W")
    
    # Step 2: Predict
    print("\n[2/4] Running predictions...")
    base_latency = profile_results['latency_ms']['total']
    base_power = profile_results['power_w']
    fps = profile_results['fps']
    
    # Test a few configurations
    test_configs = [
        {"precision": "INT8", "resolution": (640, 480), "batch_size": 1},
        {"precision": "FP16", "resolution": (640, 480), "batch_size": 1},
    ]
    
    predictions = {}
    for config in test_configs:
        pred_latency = predict_latency(
            base_latency_ms=base_latency,
            sku=args.sku,
            precision=config["precision"],
            resolution=config["resolution"],
            batch_size=config["batch_size"]
        )
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 0
        pred_power = predict_power(
            base_power_w=base_power,
            fps=pred_fps,
            precision=config["precision"],
            sku=args.sku,
            resolution=config["resolution"]
        )
        predictions[config["precision"]] = {
            "latency_ms": pred_latency,
            "power_w": pred_power,
        }
    
    print(f"  ✓ Tested {len(test_configs)} configurations")
    
    # Step 3: Optimize
    print("\n[3/4] Optimizing configuration...")
    
    def objective(knobs: ConfigKnobs) -> float:
        pred_latency = predict_latency(
            base_latency, args.sku, knobs.precision,
            knobs.resolution, knobs.batch_size
        )
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
        pred_power = predict_power(
            base_power, pred_fps, knobs.precision,
            args.sku, knobs.resolution
        )
        
        # Constraint: Power < 20W
        if pred_power > 20.0:
            return 10000.0
        
        return pred_latency + 0.1 * pred_power
    
    best_knobs = greedy_search(
        objective_fn=objective,
        initial_knobs=ConfigKnobs(
            precision="FP16",
            resolution=(640, 480),
            batch_size=1
        ),
        max_iterations=20
    )
    
    # Get predictions for best config
    best_latency = predict_latency(
        base_latency, args.sku, best_knobs.precision,
        best_knobs.resolution, best_knobs.batch_size
    )
    best_fps = 1000.0 / best_latency if best_latency > 0 else 0
    best_power = predict_power(
        base_power, best_fps, best_knobs.precision,
        args.sku, best_knobs.resolution
    )
    
    print(f"  ✓ Best config: {best_knobs.precision} @ {best_knobs.resolution}")
    print(f"  ✓ Predicted: {best_latency:.2f}ms @ {best_power:.2f}W")
    
    # Step 4: Report
    print("\n[4/4] Generating report...")
    report_gen = ReportGenerator(output_dir=str(output_dir))
    
    final_predictions = {
        "latency_ms": best_latency,
        "power_w": best_power,
    }
    
    # Thermal prediction
    thermal_model = ThermalRC(
        ambient_temp_c=25.0,
        thermal_resistance_c_per_w=0.5,
        thermal_capacitance_j_per_c=10.0,
        max_temp_c=70.0
    )
    time_to_throttle = thermal_model.time_to_throttle(best_power)
    if time_to_throttle != float('inf'):
        final_predictions["time_to_throttle_s"] = time_to_throttle
    
    report_path = report_gen.generate_report(
        profile_results=profile_results,
        predictions=final_predictions,
        optimized_config=best_knobs.to_dict()
    )
    
    print(f"  ✓ Report saved: {report_path}")
    
    # Save results JSON
    results_json = {
        "profile_results": profile_results,
        "best_config": best_knobs.to_dict(),
        "predicted_performance": {
            "latency_ms": best_latency,
            "power_w": best_power,
            "fps": best_fps,
        }
    }
    json_path = output_dir / "workflow_results.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    print(f"  ✓ Results JSON: {json_path}")
    
    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)
    
    return 0


def run_profile(args):
    """Run profiling command."""
    if not args.model:
        print("Error: --model is required", file=sys.stderr)
        return 1
    
    print(f"Profiling model: {args.model}")
    print(f"SKU: {args.sku}")
    
    profiler = PipelineProfiler(sku=args.sku)
    results = profiler.profile(
        model_path=args.model,
        video_path=args.video,
        iterations=10
    )
    
    print("\nProfile Results:")
    print(f"  Latency: {results['latency_ms']['total']:.2f} ms")
    print(f"    - Preprocess: {results['latency_ms']['preprocess']:.2f} ms")
    print(f"    - Inference: {results['latency_ms']['inference']:.2f} ms")
    print(f"    - Postprocess: {results['latency_ms']['postprocess']:.2f} ms")
    print(f"  FPS: {results['fps']:.1f}")
    print(f"  Power: {results['power_w']:.2f} W")
    print(f"  Memory: {results['memory_mb']:.2f} MB")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_path}")
    
    return 0


def run_predict(args):
    """Run prediction command."""
    print("Predict command requires profile data.")
    print("Use 'full' command or provide profile results JSON with --input")
    return 0


def run_optimize(args):
    """Run optimization command."""
    print("Optimize command requires profile data.")
    print("Use 'full' command or provide profile results JSON with --input")
    return 0


def run_optimize_model(args):
    """Run model optimization command."""
    if not args.model:
        print("Error: --model is required for optimize-model", file=sys.stderr)
        return 1
    
    # Get target metrics from args or defaults
    target_fps = getattr(args, 'target_fps', None)
    target_latency_ms = getattr(args, 'target_latency_ms', None)
    max_power_w = getattr(args, 'max_power_w', None)
    
    if not target_fps and not target_latency_ms:
        print("Error: Must specify either --target-fps or --target-latency-ms", file=sys.stderr)
        return 1
    
    print(f"Optimizing model: {args.model}")
    print(f"Target FPS: {target_fps}" if target_fps else f"Target Latency: {target_latency_ms} ms")
    print(f"Max Power: {max_power_w} W" if max_power_w else "No power constraint")
    
    output_path = args.output or str(Path(args.model).with_name(f"{Path(args.model).stem}_optimized.onnx"))
    
    try:
        results = optimize_model_for_metrics(
            model_path=args.model,
            target_fps=target_fps,
            target_latency_ms=target_latency_ms,
            max_power_w=max_power_w,
            sku=args.sku,
            output_path=output_path
        )
        
        print("\nOptimization Results:")
        print(f"  Optimized Model: {results['optimized_model_path']}")
        print(f"  Achieved FPS: {results['achieved_fps']:.1f}")
        print(f"  Latency: {results['latency_ms']:.2f} ms")
        print(f"  Power: {results['power_w']:.2f} W")
        print(f"  Precision: {results['precision']}")
        print(f"  Resolution: {results['resolution']}")
        print(f"  Status: {results['message']}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

