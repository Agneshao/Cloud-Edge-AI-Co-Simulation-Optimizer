# EdgeTwin Complete Workflow

## Overview

This document shows the complete end-to-end workflow, including how to ingest your profiling data and use it throughout the pipeline.

### When Do You Need Co-Simulation?

**Short answer: For your current scenario, co-simulation is NOT required.**

Co-simulation (Isaac Sim) is **optional** and only needed if you want to:

1. **Test in robotics scenarios**: Verify your model works in realistic scenarios (warehouse navigation, autonomous driving, manipulation tasks)
2. **Validate before deployment**: Test the optimized config in a simulated environment before deploying to real robots
3. **Scenario-based validation**: Ensure the model meets requirements in specific use cases (e.g., "Will this config work for 30-minute autonomous navigation?")

**For hardware performance optimization** (what you're doing now):
- ✅ Profile on Jetson (you have this data)
- ✅ Predict performance for different configs
- ✅ Optimize to find best latency/power tradeoff
- ✅ Generate reports

**Co-simulation adds**:
- Scenario testing (e.g., "Does this config work for 10-minute warehouse navigation?")
- End-to-end system validation (model + robot behavior)
- Long-duration testing (thermal behavior over time in realistic scenarios)

**Your workflow without co-sim:**
```
Profile → Predict → Optimize → Report
```

**With co-sim (optional):**
```
Profile → Predict → Optimize → [Co-Sim Verify] → Report
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    EdgeTwin Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA INGESTION                                               │
│     ┌──────────────────────────────────────────────┐          │
│     │  Your CSV Data → profiles_local.csv           │          │
│     │  (timestamp, power_mW, trt_latency_ms, etc.)  │          │
│     └──────────────────────────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  2. PROFILING (or use existing data)                            │
│     ┌──────────────────────────────────────────────┐          │
│     │  PipelineProfiler.profile()                  │          │
│     │  - Load ONNX model                           │          │
│     │  - Run inference on video frames              │          │
│     │  - Measure latency, power, memory             │          │
│     └──────────────────────────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  3. FEATURE ENGINEERING                                         │
│     ┌──────────────────────────────────────────────┐          │
│     │  build_features(profile_data, knobs)         │          │
│     │  - Extract numeric features                  │          │
│     │  - Normalize/encode categoricals              │          │
│     └──────────────────────────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  4. PREDICTION                                                  │
│     ┌──────────────────────────────────────────────┐          │
│     │  predict_latency()                           │          │
│     │  predict_power()                              │          │
│     │  predict_thermal()                           │          │
│     └──────────────────────────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  5. OPTIMIZATION                                                │
│     ┌──────────────────────────────────────────────┐          │
│     │  greedy_search(objective_fn)                  │          │
│     │  - Try different knob combinations            │          │
│     │  - Find best config for your constraints      │          │
│     └──────────────────────────────────────────────┘          │
│                         │                                       │
│                         ▼                                       │
│  6. REPORTING                                                   │
│     ┌──────────────────────────────────────────────┐          │
│     │  ReportGenerator.generate_report()            │          │
│     │  - HTML report with all results               │          │
│     └──────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### Step 1: Data Ingestion (Your Sample Data)

Your sample data row:
```
timestamp=1764086821, frame_id=11, engine_name=yolov8n_fp16_static,
engine_precision=FP16, engine_batch=1, engine_shape=[1,3,640,640],
end_to_end_ms=31.492104, power_mW=5686, gpu_temp_C=47.937,
ram_usage_MB=3423, ...
```

**Convert to EdgeTwin format:**

```python
# Your data transformation script
import pandas as pd

def convert_to_edgetwin_format(csv_path: str, output_path: str):
    """Convert your profiling CSV to EdgeTwin profiles_local.csv format."""
    df = pd.read_csv(csv_path)
    
    # Transform each row
    edgetwin_data = []
    for _, row in df.iterrows():
        # Extract resolution from engine_shape [1, 3, 640, 640]
        shape = eval(row['engine_shape']) if isinstance(row['engine_shape'], str) else row['engine_shape']
        height, width = shape[2], shape[3]  # Assuming NCHW format
        
        edgetwin_data.append({
            'timestamp': row['timestamp'],
            'sku': 'orin_nx',  # Determine from platform/jetson_mode
            'precision': row['engine_precision'],
            'resolution_h': height,
            'resolution_w': width,
            'batch_size': row['engine_batch'],
            'frame_skip': 0,  # From your experiment setup
            'latency_ms': row['end_to_end_ms'],
            'trt_latency_ms': row['trt_latency_ms'],
            'preprocess_ms': row['input_preprocess_ms'],
            'postprocess_ms': row['postprocess_ms'],
            'power_w': row['power_mW'] / 1000.0,  # Convert mW to W
            'power_mW': row['power_mW'],
            'gpu_temp_C': row['gpu_temp_C'],
            'gpu_util_percent': row['gpu_util_percent'],
            'ram_usage_MB': row['ram_usage_MB'],
            'gpu_mem_alloc_MB': row['gpu_mem_alloc_MB'],
            'fps': 1000.0 / row['end_to_end_ms'] if row['end_to_end_ms'] > 0 else 0,
            'model_name': row['engine_name'],
        })
    
    # Save to profiles_local.csv
    edgetwin_df = pd.DataFrame(edgetwin_data)
    edgetwin_df.to_csv(output_path, index=False)
    print(f"Converted {len(edgetwin_data)} rows to {output_path}")

# Usage
convert_to_edgetwin_format("your_profiling_data.csv", "data/jetbenchdb/profiles_local.csv")
```

---

### Step 2: Profiling (or Use Existing Data)

**Option A: Use your existing profiling data**

```python
import pandas as pd
from src.core.profile.pipeline_profiler import PipelineProfiler

# Load your existing data
df = pd.read_csv("data/jetbenchdb/profiles_local.csv")

# Use a row as baseline profile
baseline_row = df.iloc[0]
profile_results = {
    "latency_ms": {
        "preprocess": baseline_row['preprocess_ms'],
        "inference": baseline_row['trt_latency_ms'],
        "postprocess": baseline_row['postprocess_ms'],
        "total": baseline_row['latency_ms'],
    },
    "fps": baseline_row['fps'],
    "power_w": baseline_row['power_w'],
    "memory_mb": baseline_row['ram_usage_MB'],
    "sku": baseline_row['sku'],
}
```

**Option B: Run new profiling**

```python
from src.core.profile.pipeline_profiler import profile_pipeline

# Profile a model
profile_results = profile_pipeline(
    model_path="data/samples/yolov8n.onnx",
    video_path="data/samples/clip.mp4",
    sku="orin_nx",
    jetson_adapter=None,  # Use JetsonAdapter() when on real hardware
)

print(f"Latency: {profile_results['latency_ms']['total']:.2f} ms")
print(f"FPS: {profile_results['fps']:.2f}")
print(f"Power: {profile_results['power_w']:.2f} W")
```

---

### Step 3: Feature Engineering

```python
from src.core.predict.features import build_features

# From your sample data
profile_data = {
    "latency_ms": 31.492104,  # end_to_end_ms
    "power_w": 5.686,         # power_mW / 1000
    "memory_mb": 3423,        # ram_usage_MB
}

knobs = {
    "precision": "FP16",       # engine_precision
    "resolution": (640, 640),  # from engine_shape
    "batch_size": 1,          # engine_batch
    "frame_skip": 0,
}

# Build feature vector
features = build_features(profile_data, knobs)
print(f"Features: {features}")
# Output: [31.49, 5.69, 3423.0, 1.0, 640.0, 640.0, 1.0, 0.0]
#         [latency, power, memory, precision(FP16=1), h, w, batch, skip]
```

---

### Step 4: Prediction

```python
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC

# Base measurements from your profiling data
base_latency_ms = 31.492104
base_power_w = 5.686
sku = "orin_nx"  # Determine from your platform
fps = 1000.0 / base_latency_ms  # ~31.8 FPS

# Predict for different configurations
configs_to_test = [
    {"precision": "INT8", "resolution": (640, 640), "batch_size": 1},
    {"precision": "FP16", "resolution": (1280, 1280), "batch_size": 1},
    {"precision": "FP16", "resolution": (640, 640), "batch_size": 2},
]

print(f"{'Config':<40} {'Latency (ms)':<15} {'Power (W)':<15} {'FPS':<10}")
print("-" * 80)

for config in configs_to_test:
    # Predict latency
    pred_latency = predict_latency(
        base_latency_ms=base_latency_ms,
        sku=sku,
        precision=config["precision"],
        resolution=config["resolution"],
        batch_size=config["batch_size"]
    )
    
    # Predict power
    pred_power = predict_power(
        base_power_w=base_power_w,
        fps=1000.0 / pred_latency if pred_latency > 0 else 30.0,
        precision=config["precision"],
        sku=sku,
        resolution=config["resolution"],
        power_mode="15W"  # From your jetson_mode
    )
    
    # Predict thermal (time to throttle)
    thermal_model = ThermalRC(
        ambient_temp_c=25.0,
        thermal_resistance_c_per_w=0.5,
        thermal_capacitance_j_per_c=10.0,
        max_temp_c=70.0
    )
    time_to_throttle = thermal_model.time_to_throttle(pred_power)
    
    pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 0
    
    config_str = f"{config['precision']} @ {config['resolution'][0]}x{config['resolution'][1]} (batch={config['batch_size']})"
    print(f"{config_str:<40} {pred_latency:<15.2f} {pred_power:<15.2f} {pred_fps:<10.1f}")
```

---

### Step 5: Optimization

```python
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power

# Use your baseline data
base_latency_ms = 31.492104
base_power_w = 5.686
sku = "orin_nx"
fps = 31.8

# Define objective function
def objective(knobs: ConfigKnobs) -> float:
    """
    Objective: Minimize latency while keeping power under constraint.
    
    Returns: Objective score (lower is better)
    """
    # Predict latency for this configuration
    pred_latency = predict_latency(
        base_latency_ms=base_latency_ms,
        sku=sku,
        precision=knobs.precision,
        resolution=knobs.resolution,
        batch_size=knobs.batch_size
    )
    
    # Predict power
    pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
    pred_power = predict_power(
        base_power_w=base_power_w,
        fps=pred_fps,
        precision=knobs.precision,
        sku=sku,
        resolution=knobs.resolution,
        power_mode="15W"
    )
    
    # Constraint: Power must be < 15W (hard constraint)
    if pred_power > 15.0:
        return 10000.0  # Large penalty for violating constraint
    
    # Objective: Minimize latency (with small power penalty)
    return pred_latency + 0.1 * pred_power

# Run optimization
print("Running optimization...")
best_knobs = greedy_search(
    objective_fn=objective,
    initial_knobs=ConfigKnobs(
        precision="FP16",
        resolution=(640, 640),
        batch_size=1,
        frame_skip=0
    ),
    max_iterations=50
)

# Evaluate best configuration
best_latency = predict_latency(
    base_latency_ms, sku, best_knobs.precision, 
    best_knobs.resolution, best_knobs.batch_size
)
best_fps = 1000.0 / best_latency if best_latency > 0 else 0
best_power = predict_power(
    base_power_w, best_fps, best_knobs.precision, 
    sku, best_knobs.resolution, power_mode="15W"
)

print(f"\nBest Configuration Found:")
print(f"  Precision: {best_knobs.precision}")
print(f"  Resolution: {best_knobs.resolution}")
print(f"  Batch Size: {best_knobs.batch_size}")
print(f"  Frame Skip: {best_knobs.frame_skip}")
print(f"\nPredicted Performance:")
print(f"  Latency: {best_latency:.2f} ms")
print(f"  FPS: {best_fps:.1f}")
print(f"  Power: {best_power:.2f} W")
print(f"  Objective Score: {objective(best_knobs):.2f}")
```

---

### Step 6: Reporting

```python
from src.core.plan.reporter import ReportGenerator

# Generate report
report_gen = ReportGenerator()

# Profile results (from Step 2)
profile_results = {
    "latency_ms": {
        "total": 31.492104,
        "preprocess": 12.743995,
        "inference": 14.97836597,
        "postprocess": 0.0001919688657,
    },
    "fps": 31.8,
    "power_w": 5.686,
    "memory_mb": 3423,
    "sku": "orin_nx",
}

# Predictions (from Step 4)
predictions = {
    "latency_ms": best_latency,
    "power_w": best_power,
    "time_to_throttle_s": thermal_model.time_to_throttle(best_power),
}

# Optimized config (from Step 5)
optimized_config = {
    "precision": best_knobs.precision,
    "resolution": best_knobs.resolution,
    "batch_size": best_knobs.batch_size,
    "frame_skip": best_knobs.frame_skip,
}

# Generate HTML report
report_path = report_gen.generate_report(
    profile_results=profile_results,
    predictions=predictions,
    optimized_config=optimized_config
)

print(f"Report generated: {report_path}")
```

---

## Complete End-to-End Example

```python
"""
Complete workflow example using your sample data.
"""
import pandas as pd
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.plan.reporter import ReportGenerator

# Step 1: Load your profiling data
df = pd.read_csv("your_profiling_data.csv")
baseline = df.iloc[0]  # Use first row as baseline

# Extract baseline metrics
base_latency_ms = baseline['end_to_end_ms']
base_power_w = baseline['power_mW'] / 1000.0
sku = "orin_nx"  # Determine from your data
fps = 1000.0 / base_latency_ms

# Step 2: Build profile results dict
profile_results = {
    "latency_ms": {
        "total": base_latency_ms,
        "preprocess": baseline.get('input_preprocess_ms', 0),
        "inference": baseline.get('trt_latency_ms', 0),
        "postprocess": baseline.get('postprocess_ms', 0),
    },
    "fps": fps,
    "power_w": base_power_w,
    "memory_mb": baseline.get('ram_usage_MB', 0),
    "sku": sku,
}

# Step 3: Define optimization objective
def objective(knobs: ConfigKnobs) -> float:
    pred_latency = predict_latency(
        base_latency_ms, sku, knobs.precision, 
        knobs.resolution, knobs.batch_size
    )
    pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
    pred_power = predict_power(
        base_power_w, pred_fps, knobs.precision, 
        sku, knobs.resolution, power_mode="15W"
    )
    
    # Constraint: Power < 15W
    if pred_power > 15.0:
        return 10000.0
    
    # Minimize latency
    return pred_latency

# Step 4: Run optimization
print("Optimizing configuration...")
best_knobs = greedy_search(
    objective_fn=objective,
    initial_knobs=ConfigKnobs(
        precision="FP16",
        resolution=(640, 640),
        batch_size=1
    )
)

# Step 5: Get predictions for best config
best_latency = predict_latency(
    base_latency_ms, sku, best_knobs.precision,
    best_knobs.resolution, best_knobs.batch_size
)
best_fps = 1000.0 / best_latency if best_latency > 0 else 0
best_power = predict_power(
    base_power_w, best_fps, best_knobs.precision,
    sku, best_knobs.resolution, power_mode="15W"
)

# Step 6: Generate report
report_gen = ReportGenerator()
report_path = report_gen.generate_report(
    profile_results=profile_results,
    predictions={
        "latency_ms": best_latency,
        "power_w": best_power,
    },
    optimized_config=best_knobs.to_dict()
)

print(f"\n✓ Optimization complete!")
print(f"✓ Best config: {best_knobs.precision} @ {best_knobs.resolution}")
print(f"✓ Predicted: {best_latency:.2f}ms @ {best_power:.2f}W")
print(f"✓ Report: {report_path}")
```

---

## Data Flow Summary

```
Your CSV Data
    │
    ├─> Convert to profiles_local.csv format
    │
    ├─> Extract baseline metrics (latency, power, etc.)
    │
    ├─> Build features (features.py)
    │
    ├─> Predict for different configs (latency_rule.py, power.py)
    │
    ├─> Optimize (search.py) → Find best config
    │
    └─> Generate report (reporter.py) → HTML output
```

---

## Key Files Reference

- **Profiling**: `src/core/profile/pipeline_profiler.py`
- **Features**: `src/core/predict/features.py`
- **Latency Prediction**: `src/core/predict/latency_rule.py`
- **Power Prediction**: `src/core/predict/power.py`
- **Thermal Prediction**: `src/core/predict/thermal_rc.py`
- **Optimization**: `src/core/optimize/search.py`
- **Configuration**: `src/core/optimize/knobs.py`
- **Reporting**: `src/core/plan/reporter.py`

---

## Next Steps

1. **Convert your CSV data** to `profiles_local.csv` format
2. **Calibrate prediction models** using your real measurements
3. **Run optimization** to find best configurations
4. **Generate reports** for different scenarios
5. **Integrate with API/CLI** for production use

