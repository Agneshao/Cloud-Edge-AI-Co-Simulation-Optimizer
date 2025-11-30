"""FastAPI server for EdgeTwin."""

from fastapi import FastAPI, HTTPException
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .schemas import (
    ProfileRequest, ProfileResponse,
    PredictRequest, PredictResponse,
    OptimizeRequest, OptimizeResponse,
    ModelOptimizeRequest, ModelOptimizeResponse
)
from src.core.profile.pipeline_profiler import PipelineProfiler
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search

app = FastAPI(
    title="EdgeTwin API",
    description="Hardware-aware co-simulation platform for robotics AI",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "EdgeTwin API", "version": "0.1.0"}


@app.post("/profile", response_model=ProfileResponse)
async def profile(request: ProfileRequest):
    """
    Profile a model on simulated Jetson hardware.
    """
    try:
        profiler = PipelineProfiler(sku=request.sku)
        results = profiler.profile(
            model_path=request.model_path,
            video_path=request.video_path,
            iterations=request.iterations
        )
        
        return ProfileResponse(
            latency_ms=results['latency_ms']['total'],
            power_w=results['power_w'],
            memory_mb=results['memory_mb'],
            fps=results['fps'],
            sku=request.sku
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict performance for given configuration.
    """
    try:
        # Predict latency
        pred_latency = predict_latency(
            base_latency_ms=request.base_latency_ms,
            sku=request.sku,
            precision=request.precision,
            resolution=request.resolution,
            batch_size=request.batch_size
        )
        
        # Predict power
        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else request.fps
        pred_power = predict_power(
            base_power_w=request.base_power_w,
            fps=pred_fps,
            precision=request.precision,
            sku=request.sku,
            resolution=request.resolution
        )
        
        # Predict thermal (time to throttle)
        thermal_model = ThermalRC(
            ambient_temp_c=25.0,
            thermal_resistance_c_per_w=0.5,
            thermal_capacitance_j_per_c=10.0,
            max_temp_c=70.0
        )
        time_to_throttle = thermal_model.time_to_throttle(pred_power)
        
        return PredictResponse(
            latency_ms=pred_latency,
            power_w=pred_power,
            time_to_throttle_s=time_to_throttle if time_to_throttle != float('inf') else -1.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Optimize configuration given constraints.
    """
    try:
        profile_results = request.profile_results
        constraints = request.constraints
        weights = request.objective_weights or {"latency": 1.0, "power": 0.1}
        
        # Extract baseline metrics
        latency_ms = profile_results.get("latency_ms", {})
        if isinstance(latency_ms, dict):
            base_latency_ms = latency_ms.get("total", 50.0)
        else:
            base_latency_ms = float(latency_ms)
        
        base_power_w = float(profile_results.get("power_w", 10.0))
        sku = profile_results.get("sku", "orin_super")
        
        # Define objective function
        def objective(knobs: ConfigKnobs) -> float:
            pred_latency = predict_latency(
                base_latency_ms, sku, knobs.precision,
                knobs.resolution, knobs.batch_size
            )
            pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
            pred_power = predict_power(
                base_power_w, pred_fps, knobs.precision,
                sku, knobs.resolution
            )
            
            # Apply constraints
            max_power = constraints.get("max_power_w", 20.0)
            max_latency = constraints.get("max_latency_ms", 100.0)
            
            if pred_power > max_power:
                return 10000.0  # Hard constraint violation
            if pred_latency > max_latency:
                return 10000.0  # Hard constraint violation
            
            # Weighted objective
            return weights["latency"] * pred_latency + weights["power"] * pred_power
        
        # Run optimization
        best_knobs = greedy_search(
            objective_fn=objective,
            initial_knobs=ConfigKnobs(),
            max_iterations=50
        )
        
        # Get predictions for best config
        best_latency = predict_latency(
            base_latency_ms, sku, best_knobs.precision,
            best_knobs.resolution, best_knobs.batch_size
        )
        best_fps = 1000.0 / best_latency if best_latency > 0 else 0
        best_power = predict_power(
            base_power_w, best_fps, best_knobs.precision,
            sku, best_knobs.resolution
        )
        
        return OptimizeResponse(
            optimized_config=best_knobs.to_dict(),
            predicted_performance={
                "latency_ms": best_latency,
                "power_w": best_power,
                "fps": best_fps
            },
            objective_value=objective(best_knobs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/optimize-model", response_model=ModelOptimizeResponse)
async def optimize_model(request: ModelOptimizeRequest):
    """
    Optimize model file to meet target metrics (FPS, latency, power).
    
    This endpoint:
    1. Takes an ONNX model file
    2. Optimizes it (quantization, precision conversion) based on target metrics
    3. Returns optimized model path and performance metrics
    """
    try:
        from src.core.optimize.model_converter import optimize_model_for_metrics
        
        results = optimize_model_for_metrics(
            model_path=request.model_path,
            target_fps=request.target_fps,
            target_latency_ms=request.target_latency_ms,
            max_power_w=request.max_power_w,
            sku=request.sku,
            output_path=request.output_path
        )
        
        return ModelOptimizeResponse(**results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model optimization failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

