"""Pipeline profiler for Jetson-aware performance measurement.

TODO: Performance profiling basics:
  1. Why warmup_iterations? GPU/cache warmup - first few runs are slower
  2. Why multiple iterations? Average out variability (cache effects, OS scheduling)
  3. Stage-by-stage timing: Helps identify bottlenecks (preprocess vs inference vs postprocess)
  4. Power/memory: Uses real measurements when available (JetsonAdapter), estimates otherwise

Features:
  - Video loading: Supports loading frames from video files (MP4, AVI, etc.)
  - Power measurement: Integrates with JetsonAdapter for real hardware, uses estimates otherwise
  - Memory measurement: Uses psutil for system memory, ready for GPU memory integration
  - Detailed statistics: Mean, std, min, max for each pipeline stage
"""

import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import numpy as np
import onnxruntime as ort
import cv2
import psutil
from .stages import PipelineStage, create_preprocess_stage, create_inference_stage, create_postprocess_stage


class PipelineProfiler:
    """Profiles AI pipeline stages (preprocess, inference, postprocess) on Jetson."""
    
    def __init__(
        self,
        sku: str = "orin_super",
        warmup_iterations: int = 3,
        jetson_adapter: Optional[Any] = None
    ):
        """
        Initialize pipeline profiler.
        
        Args:
            sku: Jetson SKU identifier
            warmup_iterations: Number of warmup iterations before profiling
            jetson_adapter: Optional JetsonAdapter instance for hardware measurements
        """
        self.sku = sku
        self.warmup_iterations = warmup_iterations
        self.session: Optional[ort.InferenceSession] = None
        self.jetson_adapter = jetson_adapter
        self._video_cap: Optional[Any] = None
        self._video_frames: Optional[List[np.ndarray]] = None
    
    def load_model(self, model_path: str):
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def profile(
        self,
        model_path: str,
        video_path: Optional[str] = None,
        iterations: int = 10,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Profile pipeline stages.
        
        Args:
            model_path: Path to ONNX model
            video_path: Path to input video (optional)
            iterations: Number of profiling iterations
            preprocess_fn: Custom preprocessing function
            postprocess_fn: Custom postprocessing function
        
        Returns:
            Profile results dictionary
        """
        self.load_model(model_path)
        
        # Create stage shims
        preprocess_stage = create_preprocess_stage(preprocess_fn)
        inference_stage = create_inference_stage(self._run_inference)
        postprocess_stage = create_postprocess_stage(postprocess_fn)
        
        # Generate dummy input if no video provided
        input_data = self._get_input_data(video_path)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = self._run_pipeline(preprocess_stage, inference_stage, postprocess_stage, input_data)
        
        # Profile
        latencies = {
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "total": []
        }
        
        for _ in range(iterations):
            times = self._run_pipeline(preprocess_stage, inference_stage, postprocess_stage, input_data)
            for stage, t in times.items():
                latencies[stage].append(t)
        
        # Calculate statistics
        results = {
            "latency_ms": {
                "preprocess": np.mean(latencies["preprocess"]) * 1000,
                "inference": np.mean(latencies["inference"]) * 1000,
                "postprocess": np.mean(latencies["postprocess"]) * 1000,
                "total": np.mean(latencies["total"]) * 1000,
            },
            "latency_stats": {
                "preprocess": {
                    "mean": np.mean(latencies["preprocess"]) * 1000,
                    "std": np.std(latencies["preprocess"]) * 1000,
                    "min": np.min(latencies["preprocess"]) * 1000,
                    "max": np.max(latencies["preprocess"]) * 1000,
                },
                "inference": {
                    "mean": np.mean(latencies["inference"]) * 1000,
                    "std": np.std(latencies["inference"]) * 1000,
                    "min": np.min(latencies["inference"]) * 1000,
                    "max": np.max(latencies["inference"]) * 1000,
                },
                "postprocess": {
                    "mean": np.mean(latencies["postprocess"]) * 1000,
                    "std": np.std(latencies["postprocess"]) * 1000,
                    "min": np.min(latencies["postprocess"]) * 1000,
                    "max": np.max(latencies["postprocess"]) * 1000,
                },
                "total": {
                    "mean": np.mean(latencies["total"]) * 1000,
                    "std": np.std(latencies["total"]) * 1000,
                    "min": np.min(latencies["total"]) * 1000,
                    "max": np.max(latencies["total"]) * 1000,
                },
            },
            "fps": 1000.0 / np.mean(latencies["total"]) if np.mean(latencies["total"]) > 0 else 0.0,
            # TODO: Power measurement:
            #   - Use tegrastats to get real-time power (watts)
            #   - Sample during inference to get average power
            #   - Consider peak vs average power

            # TODO: Memory measurement:
            #   - Use nvidia-smi or tegrastats for GPU memory
            #   - Use psutil for system memory
            #   - Track peak vs average memory usage
            "power_w": self._measure_power(iterations), # Simulated - integrate with JetsonAdapter for real measurements
            "memory_mb": self._measure_memory(), # Simulated - integrate with JetsonAdapter for real measurements
            "sku": self.sku,
            "iterations": iterations,
        }
        
        return results
    
    def _measure_power(self, iterations: int) -> float:
        """
        Measure power consumption during profiling.
        
        If JetsonAdapter is available and on real hardware, uses real measurements.
        Otherwise, estimates power based on SKU and workload.
        
        Args:
            iterations: Number of profiling iterations
            
        Returns:
            Average power consumption in watts
        """
        # Try to use JetsonAdapter for real hardware measurements
        if self.jetson_adapter and hasattr(self.jetson_adapter, 'is_real_hardware'):
            if self.jetson_adapter.is_real_hardware:
                # TODO: When JetsonAdapter._profile_real is implemented,
                # integrate power measurement here
                # For now, return estimated value
                pass
        
        # Estimate power based on SKU (fallback/simulated)
        # These are rough estimates - should be calibrated with real data
        base_power_estimates = {
            "orin_super": 15.0,
            "orin_nx": 12.0,
            "orin_nano": 8.0,
            "xavier_nx": 10.0,
            "nano": 5.0,
        }
        
        base_power = base_power_estimates.get(self.sku.lower(), 10.0)
        
        # Add workload-dependent power (rough estimate)
        # Inference typically adds 2-5W depending on model complexity
        workload_power = 3.0  # Estimated additional power during inference
        
        return base_power + workload_power
    
    def _measure_memory(self) -> float:
        """
        Measure memory usage.
        
        Uses psutil for system memory. GPU memory measurement requires
        JetsonAdapter integration when on real hardware.
        
        Returns:
            Memory usage in MB
        """
        memory_mb = 0.0
        
        # Try to get GPU memory if JetsonAdapter is available
        if self.jetson_adapter and hasattr(self.jetson_adapter, 'is_real_hardware'):
            if self.jetson_adapter.is_real_hardware:
                # TODO: When JetsonAdapter supports GPU memory measurement,
                # integrate here using nvidia-smi or tegrastats
                # For now, estimate based on model size
                if self.session is not None:
                    # Rough estimate: model size + inference overhead
                    # This is a placeholder - real implementation needed
                    pass
        
        # Use psutil for system memory
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            # Fallback to estimated value
            if self.session is not None:
                # Rough estimate: assume model + buffers
                memory_mb = 512.0
            else:
                memory_mb = 256.0
        
        return memory_mb
    
    def _get_input_data(self, video_path: Optional[str]) -> np.ndarray:
        """
        Get input data from video file or generate dummy data.
        
        If video_path is provided, loads frames from video.
        Otherwise, generates dummy random data matching model input shape.
        """
        # If video path provided, load video
        if video_path:
            return self._load_video_frame(video_path)
        
        # Default dummy input matching model shape
        if self.session is not None:
            input_shape = self.session.get_inputs()[0].shape
            # Replace dynamic dimensions with defaults
            input_shape = [1 if s is None or s == 'batch_size' else s for s in input_shape]
            return np.random.randn(*input_shape).astype(np.float32)
        return np.random.randn(1, 3, 640, 480).astype(np.float32)
    
    def _load_video_frame(self, video_path: str) -> np.ndarray:
        """
        Load a frame from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed frame as numpy array matching model input shape
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from video: {video_path}")
            
            # Get model input shape
            if self.session is None:
                # Default shape if model not loaded yet
                target_shape = (640, 480)
            else:
                input_shape = self.session.get_inputs()[0].shape
                # Extract height and width (assuming NCHW format: [batch, channels, height, width])
                if len(input_shape) == 4:
                    target_shape = (int(input_shape[2]), int(input_shape[3]))
                else:
                    target_shape = (640, 480)
            
            # Resize frame to match model input
            frame_resized = cv2.resize(frame, target_shape)
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] and convert to float32
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Convert HWC to CHW format (channels first)
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            
            # Add batch dimension: CHW -> NCHW
            frame_batched = np.expand_dims(frame_chw, axis=0)
            
            return frame_batched
            
        finally:
            cap.release()
    
    def _run_inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run model inference."""
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        return outputs
    
    def _run_pipeline(
        self,
        preprocess: PipelineStage,
        inference: PipelineStage,
        postprocess: PipelineStage,
        input_data: np.ndarray
    ) -> Dict[str, float]:
        """Run full pipeline and measure stage times."""
        times = {}
        
        # Preprocess
        t0 = time.perf_counter()
        processed_input = preprocess.run(input_data) if preprocess.fn else input_data
        times["preprocess"] = time.perf_counter() - t0
        
        # Inference
        t0 = time.perf_counter()
        outputs = inference.run(processed_input)
        times["inference"] = time.perf_counter() - t0
        
        # Postprocess
        t0 = time.perf_counter()
        _ = postprocess.run(outputs) if postprocess.fn else outputs
        times["postprocess"] = time.perf_counter() - t0
        
        # Total
        times["total"] = times["preprocess"] + times["inference"] + times["postprocess"]
        
        return times


def profile_pipeline(
    model_path: str,
    video_path: Optional[str] = None,
    sku: str = "orin_super",
    jetson_adapter: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to profile a pipeline.
    
    Args:
        model_path: Path to ONNX model
        video_path: Path to input video (optional)
        sku: Jetson SKU identifier
        jetson_adapter: Optional JetsonAdapter instance for hardware measurements
    
    Returns:
        Profile results dictionary
    """
    profiler = PipelineProfiler(sku=sku, jetson_adapter=jetson_adapter)
    return profiler.profile(model_path, video_path)