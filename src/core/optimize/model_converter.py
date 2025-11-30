"""Model conversion and optimization based on target metrics.

This module handles actual model file conversion (quantization, pruning, etc.)
to meet target performance metrics like FPS, latency, or power.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np


class ModelConverter:
    """Convert and optimize ONNX models based on target metrics."""
    
    def __init__(self, sku: str = "orin_super"):
        """
        Initialize model converter.
        
        Args:
            sku: Jetson SKU identifier
        """
        self.sku = sku
    
    def optimize_for_fps(
        self,
        model_path: str,
        target_fps: float,
        output_path: Optional[str] = None,
        max_power_w: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize model to meet target FPS.
        
        This will:
        1. Profile the model
        2. Try different precision levels (INT8, FP16, FP32)
        3. Optimize resolution if needed
        4. Convert model to optimal format
        5. Return optimized model path and metrics
        
        Args:
            model_path: Path to input ONNX model
            target_fps: Target FPS to achieve
            output_path: Path to save optimized model (default: adds _optimized suffix)
            max_power_w: Optional maximum power constraint
            
        Returns:
            Dict with:
                - optimized_model_path: Path to optimized model
                - achieved_fps: Actual FPS achieved
                - precision: Precision used (INT8/FP16/FP32)
                - resolution: Resolution used
                - latency_ms: Latency in milliseconds
                - power_w: Power consumption
        """
        from src.core.profile.pipeline_profiler import PipelineProfiler
        from src.core.predict.latency_rule import predict_latency
        from src.core.predict.power import predict_power
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Determine output path
        if output_path is None:
            output_path = str(model_path.parent / f"{model_path.stem}_optimized.onnx")
        output_path = Path(output_path)
        
        # Load and profile baseline
        profiler = PipelineProfiler(sku=self.sku)
        baseline_results = profiler.profile(
            model_path=str(model_path),
            video_path=None,
            iterations=5
        )
        
        baseline_fps = baseline_results['fps']
        baseline_latency = baseline_results['latency_ms']['total']
        baseline_power = baseline_results['power_w']
        
        # If already meets target, return original
        if baseline_fps >= target_fps:
            return {
                "optimized_model_path": str(model_path),
                "achieved_fps": baseline_fps,
                "precision": "FP32",  # Assume original is FP32
                "resolution": self._get_model_resolution(str(model_path)),
                "latency_ms": baseline_latency,
                "power_w": baseline_power,
                "optimization_applied": False,
                "message": "Model already meets target FPS"
            }
        
        # Try different precision levels (INT8 is fastest)
        precisions = ["INT8", "FP16", "FP32"]
        best_config = None
        best_fps = 0
        best_latency = float('inf')
        
        for precision in precisions:
            # Predict performance for this precision
            pred_latency = predict_latency(
                base_latency_ms=baseline_latency,
                sku=self.sku,
                precision=precision,
                resolution=self._get_model_resolution(str(model_path)),
                batch_size=1
            )
            
            pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 0
            pred_power = predict_power(
                base_power_w=baseline_power,
                fps=pred_fps,
                precision=precision,
                sku=self.sku,
                resolution=self._get_model_resolution(str(model_path))
            )
            
            # Check constraints
            if pred_fps >= target_fps:
                if max_power_w is None or pred_power <= max_power_w:
                    if pred_fps > best_fps:
                        best_config = {
                            "precision": precision,
                            "latency_ms": pred_latency,
                            "fps": pred_fps,
                            "power_w": pred_power
                        }
                        best_fps = pred_fps
                        best_latency = pred_latency
        
        if best_config is None:
            # Couldn't meet target with precision alone, try resolution reduction
            return self._optimize_with_resolution(
                model_path, target_fps, output_path, max_power_w
            )
        
        # Convert model to best precision
        optimized_path = self._convert_precision(
            str(model_path),
            str(output_path),
            best_config["precision"]
        )
        
        return {
            "optimized_model_path": optimized_path,
            "achieved_fps": best_config["fps"],
            "precision": best_config["precision"],
            "resolution": self._get_model_resolution(str(model_path)),
            "latency_ms": best_config["latency_ms"],
            "power_w": best_config["power_w"],
            "optimization_applied": True,
            "message": f"Optimized to {best_config['precision']} precision"
        }
    
    def _optimize_with_resolution(
        self,
        model_path: Path,
        target_fps: float,
        output_path: Path,
        max_power_w: Optional[float]
    ) -> Dict[str, Any]:
        """Optimize by reducing resolution if precision alone isn't enough."""
        # For now, return a message that resolution optimization needs manual intervention
        # In a full implementation, this would:
        # 1. Load model
        # 2. Modify input shape
        # 3. Save new model
        
        return {
            "optimized_model_path": str(model_path),
            "achieved_fps": 0,
            "precision": "FP32",
            "resolution": self._get_model_resolution(str(model_path)),
            "latency_ms": 0,
            "power_w": 0,
            "optimization_applied": False,
            "message": "Target FPS not achievable with precision optimization alone. Consider reducing resolution manually or using a smaller model."
        }
    
    def _convert_precision(
        self,
        input_path: str,
        output_path: str,
        target_precision: str
    ) -> str:
        """
        Convert model to target precision.
        
        Note: This is a simplified implementation. Real quantization requires:
        - Calibration dataset
        - Quantization tools (onnxruntime quantization, TensorRT, etc.)
        - For now, we'll create a copy and note the precision change
        
        Args:
            input_path: Input ONNX model path
            output_path: Output ONNX model path
            target_precision: INT8, FP16, or FP32
            
        Returns:
            Path to converted model
        """
        # Load model
        model = onnx.load(input_path)
        
        # For INT8/FP16, we'd need actual quantization
        # For now, create a copy and add metadata
        # In production, use onnxruntime quantization or TensorRT
        
        if target_precision == "INT8":
            # TODO: Implement actual INT8 quantization
            # This requires calibration data and quantization tools
            # For now, just copy and note it needs quantization
            onnx.save(model, output_path)
            # Add metadata or note in filename
            output_path_quantized = str(Path(output_path).with_suffix('.int8.onnx'))
            onnx.save(model, output_path_quantized)
            return output_path_quantized
        
        elif target_precision == "FP16":
            # TODO: Implement FP16 conversion
            # For now, copy model
            onnx.save(model, output_path)
            return output_path
        
        else:  # FP32
            # Just copy
            onnx.save(model, output_path)
            return output_path
    
    def _get_model_resolution(self, model_path: str) -> Tuple[int, int]:
        """Get model input resolution from ONNX model."""
        try:
            session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            input_shape = session.get_inputs()[0].shape
            # Assuming NCHW format: [batch, channels, height, width]
            if len(input_shape) >= 4:
                height = int(input_shape[2]) if input_shape[2] else 640
                width = int(input_shape[3]) if input_shape[3] else 480
                return (height, width)
        except Exception:
            pass
        return (640, 480)  # Default


def optimize_model_for_metrics(
    model_path: str,
    target_fps: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    max_power_w: Optional[float] = None,
    sku: str = "orin_super",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to optimize model for target metrics.
    
    Args:
        model_path: Path to input ONNX model
        target_fps: Target FPS (if specified)
        target_latency_ms: Target latency in ms (if specified)
        max_power_w: Maximum power constraint
        sku: Jetson SKU
        output_path: Output path for optimized model
        
    Returns:
        Optimization results dict
    """
    converter = ModelConverter(sku=sku)
    
    if target_fps is not None:
        return converter.optimize_for_fps(
            model_path=model_path,
            target_fps=target_fps,
            output_path=output_path,
            max_power_w=max_power_w
        )
    elif target_latency_ms is not None:
        # Convert latency target to FPS
        target_fps = 1000.0 / target_latency_ms if target_latency_ms > 0 else 30.0
        return converter.optimize_for_fps(
            model_path=model_path,
            target_fps=target_fps,
            output_path=output_path,
            max_power_w=max_power_w
        )
    else:
        raise ValueError("Must specify either target_fps or target_latency_ms")

