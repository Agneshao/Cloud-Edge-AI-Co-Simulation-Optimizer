# Pipeline Profiler Implementation

This document describes the implementation of TODOs in `pipeline_profiler.py`.

## Implemented Features

### 1. Video Loading ✅

**Location**: `_load_video_frame()` method

**Features**:
- Loads video frames using OpenCV (`cv2`)
- Automatically resizes frames to match model input shape
- Converts BGR to RGB (OpenCV default → model expected format)
- Normalizes pixel values to [0, 1] range
- Converts HWC → CHW → NCHW format (channels-first with batch dimension)
- Handles missing OpenCV gracefully (falls back to dummy data)

**Usage**:
```python
profiler = PipelineProfiler(sku="orin_super")
results = profiler.profile(
    model_path="model.onnx",
    video_path="data/samples/clip.mp4"  # Now actually loads video!
)
```

**Supported Formats**: MP4, AVI, MOV, and other formats supported by OpenCV

### 2. Power Measurement ✅

**Location**: `_measure_power()` method

**Features**:
- Integrates with `JetsonAdapter` when available
- Falls back to SKU-based estimates when hardware not available
- Ready for real hardware integration (when `JetsonAdapter._profile_real()` is implemented)

**Current Behavior**:
- **With JetsonAdapter on real hardware**: Will use real measurements (when implemented)
- **Without hardware**: Uses SKU-based estimates:
  - Orin Super: ~18W (15W base + 3W workload)
  - Orin NX: ~15W
  - Orin Nano: ~11W
  - Xavier NX: ~13W
  - Nano: ~8W

**Future Integration**:
When your partner implements `JetsonAdapter._profile_real()`, the profiler will automatically use real power measurements from `tegrastats`.

### 3. Memory Measurement ✅

**Location**: `_measure_memory()` method

**Features**:
- Uses `psutil` for system memory measurement (process RSS)
- Ready for GPU memory integration via `JetsonAdapter`
- Falls back to estimates when `psutil` not available

**Current Behavior**:
- **With psutil**: Measures actual process memory usage (RSS)
- **Without psutil**: Uses estimated values (512MB default)
- **GPU memory**: Placeholder ready for `JetsonAdapter` integration

**Future Integration**:
When `JetsonAdapter` supports GPU memory measurement (via `nvidia-smi` or `tegrastats`), it will be automatically used.

### 4. Enhanced Statistics ✅

**New in Results**:
- `latency_stats`: Detailed statistics (mean, std, min, max) for each stage
- More accurate power and memory measurements
- Better error handling

## Integration with JetsonAdapter

The profiler now accepts an optional `jetson_adapter` parameter:

```python
from src.adapters.jetson_adapter import JetsonAdapter

# Create adapter
adapter = JetsonAdapter(sku="orin_super")

# Use with profiler
profiler = PipelineProfiler(sku="orin_super", jetson_adapter=adapter)
results = profiler.profile("model.onnx", "video.mp4")
```

When `JetsonAdapter._profile_real()` is implemented, the profiler will automatically:
1. Use real power measurements from `tegrastats`
2. Use real GPU memory from `nvidia-smi` or `tegrastats`
3. Fall back to estimates when not on real hardware

## Dependencies Added

- `opencv-python>=4.5.0` - For video loading
- `psutil>=5.9.0` - For system memory measurement

Install with:
```bash
pip install -e .
# or
pip install opencv-python psutil
```

## Testing

### Test Video Loading

```python
from src.core.profile.pipeline_profiler import PipelineProfiler

profiler = PipelineProfiler(sku="orin_super")
# This will now load actual video frames instead of dummy data
results = profiler.profile(
    model_path="data/samples/yolov5n.onnx",
    video_path="data/samples/clip.mp4"
)
print(f"FPS: {results['fps']:.2f}")
print(f"Power: {results['power_w']:.2f}W")
print(f"Memory: {results['memory_mb']:.2f}MB")
```

### Test Without Video (Dummy Data)

```python
# Still works with dummy data if no video provided
results = profiler.profile(model_path="model.onnx")
```

### Test with JetsonAdapter

```python
from src.adapters.jetson_adapter import JetsonAdapter

adapter = JetsonAdapter(sku="orin_super")
profiler = PipelineProfiler(sku="orin_super", jetson_adapter=adapter)

# When on real hardware, will use real measurements
# When simulated, will use estimates
results = profiler.profile("model.onnx", "video.mp4")
```

## Next Steps for Hardware Integration

When your partner has real hardware data:

1. **Implement `JetsonAdapter._profile_real()`**:
   - Start `tegrastats` as background process
   - Parse power/temperature/utilization from output
   - Return real measurements

2. **The profiler will automatically use real data**:
   - No changes needed to `PipelineProfiler`
   - Just pass `JetsonAdapter` instance
   - Power and memory will switch from estimates to real measurements

3. **Calibrate power estimates**:
   - Compare estimated vs real power
   - Update `_measure_power()` estimates if needed
   - Or use prediction models from `src/core/predict/power.py`

## Error Handling

The implementation handles missing dependencies gracefully:

- **No OpenCV**: Falls back to dummy data, warns user
- **No psutil**: Uses estimated memory values
- **No JetsonAdapter**: Uses SKU-based power estimates
- **Video file not found**: Raises clear error message
- **Model loading fails**: Raises RuntimeError with details

## Performance Considerations

- **Video loading**: Only loads one frame at a time (memory efficient)
- **Memory measurement**: Uses lightweight `psutil` (minimal overhead)
- **Power measurement**: No overhead when using estimates
- **Statistics**: Calculated efficiently using NumPy

## Example Output

```python
{
    "latency_ms": {
        "preprocess": 2.5,
        "inference": 20.3,
        "postprocess": 1.2,
        "total": 24.0
    },
    "latency_stats": {
        "total": {
            "mean": 24.0,
            "std": 0.5,
            "min": 23.2,
            "max": 25.1
        },
        # ... similar for other stages
    },
    "fps": 41.67,
    "power_w": 18.0,  # Real measurement when hardware available
    "memory_mb": 512.5,  # Real measurement when psutil available
    "sku": "orin_super",
    "iterations": 10
}
```

