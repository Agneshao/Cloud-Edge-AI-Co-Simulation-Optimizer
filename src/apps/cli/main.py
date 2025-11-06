"""CLI that wires profile→predict→optimize→report.

TODO: CLI design patterns:
  1. Argument parsing: Use argparse (current) or click library
  2. Command structure: Subcommands (profile, predict, optimize, full)
  3. Error handling: Validate inputs, provide helpful error messages
  4. Progress indicators: Show progress for long-running operations
  5. Logging: Use Python logging module for structured logs
  6. Configuration: Support config files (YAML, JSON) for defaults
"""

import argparse
from pathlib import Path
import sys


def main():
    """
    Main CLI entry point.
    
    TODO: Implementation steps:
      1. Import modules: from src.core.profile import PipelineProfiler
      2. Call profiler: profiler = PipelineProfiler(sku=args.sku)
      3. Process results: results = profiler.profile(args.model, args.video)
      4. Chain operations: profile → predict → optimize → report
      5. Error handling: Try/except with helpful messages
      6. Output: Save results to files, print to console
    """
    parser = argparse.ArgumentParser(
        description="EdgeTwin: Hardware-aware co-simulation for robotics AI"
    )
    
    parser.add_argument(
        "command",
        choices=["profile", "predict", "optimize", "full"],
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
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    if args.command == "full":
        print("Running full workflow: profile → predict → optimize → report")
        # TODO: Full workflow implementation:
        #   1. Profile: PipelineProfiler().profile(args.model, args.video)
        #   2. Predict: Use predict_power(), predict_latency() with profile results
        #   3. Optimize: Use greedy_search() with objective function
        #   4. Report: Use ReportGenerator().generate_report()
        #   5. Save: Write results to args.output directory
        print("Workflow not yet implemented. Use individual commands.")
    
    elif args.command == "profile":
        print(f"Profiling model: {args.model}")
        # TODO: Profile command:
        #   from src.core.profile import PipelineProfiler
        #   profiler = PipelineProfiler(sku=args.sku)
        #   results = profiler.profile(args.model, args.video)
        #   print(f"Latency: {results['latency_ms']}ms, FPS: {results['fps']}")
        print("Profile command not yet fully implemented.")
    
    elif args.command == "predict":
        # TODO: Predict command:
        #   Load profile results, then call predict_power(), predict_latency()
        #   Show predicted vs actual (if available)
        print("Predict command not yet fully implemented.")
    
    elif args.command == "optimize":
        # TODO: Optimize command:
        #   Define objective function (combine latency, power, etc.)
        #   Call greedy_search() with objective function
        #   Show best configuration found
        print("Optimize command not yet fully implemented.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

