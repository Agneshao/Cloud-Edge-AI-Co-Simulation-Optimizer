"""Pydantic schemas for API requests and responses."""

from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class ProfileRequest(BaseModel):
    """Request to profile a model."""
    model_path: str = Field(..., description="Path to ONNX model")
    video_path: Optional[str] = Field(None, description="Path to input video")
    sku: str = Field("orin_super", description="Jetson SKU identifier")
    iterations: int = Field(10, description="Number of profiling iterations")


class ProfileResponse(BaseModel):
    """Response from profiling."""
    latency_ms: float
    power_w: float
    memory_mb: float
    fps: float
    sku: str


class PredictRequest(BaseModel):
    """Request to predict performance."""
    base_latency_ms: float
    base_power_w: float
    sku: str
    precision: str = Field("FP16", description="INT8, FP16, or FP32")
    resolution: Tuple[int, int] = Field((640, 480), description="(height, width)")
    batch_size: int = Field(1, description="Batch size")
    fps: float = Field(30.0, description="Target FPS")


class PredictResponse(BaseModel):
    """Response from prediction."""
    latency_ms: float
    power_w: float
    time_to_throttle_s: float


class OptimizeRequest(BaseModel):
    """Request to optimize configuration."""
    profile_results: Dict[str, Any]
    constraints: Dict[str, Any]
    objective_weights: Optional[Dict[str, float]] = None


class OptimizeResponse(BaseModel):
    """Response from optimization."""
    optimized_config: Dict[str, Any]
    predicted_performance: Dict[str, Any]
    objective_value: float

