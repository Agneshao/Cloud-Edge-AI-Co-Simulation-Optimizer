"""E2E test for full workflow: profile→predict→optimize→report."""

import pytest
from pathlib import Path
import sys
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.profile.pipeline_profiler import PipelineProfiler
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.plan.reporter import ReportGenerator


def create_dummy_onnx_model(output_path: str):
    """Create a minimal dummy ONNX model for testing."""
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model: input -> add -> output
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, 3, 640, 480]
        )
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 3, 640, 480]
        )
        
        # Create a simple identity-like node
        node = helper.make_node(
            'Add',
            inputs=['input', 'input'],
            outputs=['output']
        )
        
        graph = helper.make_graph(
            [node],
            'test_graph',
            [input_tensor],
            [output_tensor]
        )
        
        model = helper.make_model(graph)
        onnx.save(model, output_path)
        return True
    except ImportError:
        # If onnx not available, create a dummy file
        Path(output_path).write_bytes(b'dummy onnx model')
        return False


def test_full_workflow():
    """Test the complete EdgeTwin workflow."""
    # Create temporary model file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
        model_path = tmp_model.name
        create_dummy_onnx_model(model_path)
    
    try:
        # Step 1: Profile
        profiler = PipelineProfiler(sku="orin_super")
        profile_results = profiler.profile(
            model_path=model_path,
            video_path=None,  # Use dummy data
            iterations=5
        )
        
        assert "latency_ms" in profile_results
        assert "power_w" in profile_results
        assert "fps" in profile_results
        assert profile_results["latency_ms"]["total"] > 0
        
        # Step 2: Predict
        base_latency = profile_results["latency_ms"]["total"]
        base_power = profile_results["power_w"]
        sku = profile_results["sku"]
        
        pred_latency = predict_latency(
            base_latency_ms=base_latency,
            sku=sku,
            precision="FP16",
            resolution=(640, 480),
            batch_size=1
        )
        
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
        pred_power = predict_power(
            base_power_w=base_power,
            fps=pred_fps,
            precision="FP16",
            sku=sku,
            resolution=(640, 480)
        )
        
        assert pred_latency > 0
        assert pred_power > 0
        
        # Step 3: Optimize
        def objective(knobs: ConfigKnobs) -> float:
            pred_lat = predict_latency(
                base_latency, sku, knobs.precision,
                knobs.resolution, knobs.batch_size
            )
            pred_f = 1000.0 / pred_lat if pred_lat > 0 else 30.0
            pred_p = predict_power(
                base_power, pred_f, knobs.precision,
                sku, knobs.resolution
            )
            if pred_p > 20.0:
                return 10000.0
            return pred_lat + 0.1 * pred_p
        
        best_knobs = greedy_search(
            objective_fn=objective,
            initial_knobs=ConfigKnobs(),
            max_iterations=10
        )
        
        assert best_knobs is not None
        assert best_knobs.precision in ["INT8", "FP16", "FP32"]
        
        # Step 4: Generate report
        report_gen = ReportGenerator()
        report_path = report_gen.generate_report(
            profile_results=profile_results,
            predictions={
                "latency_ms": pred_latency,
                "power_w": pred_power,
            },
            optimized_config=best_knobs.to_dict()
        )
        
        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".html"
        
        # Verify report content
        report_content = Path(report_path).read_text()
        assert "EdgeTwin Performance Report" in report_content
        assert str(pred_latency) in report_content or f"{pred_latency:.2f}" in report_content
        
    finally:
        # Cleanup
        if Path(model_path).exists():
            Path(model_path).unlink()

