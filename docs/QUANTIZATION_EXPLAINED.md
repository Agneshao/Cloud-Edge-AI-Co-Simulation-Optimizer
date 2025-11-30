# Model Quantization Explained

## What is Quantization?

Quantization is the process of converting a model from higher precision (FP32, FP16) to lower precision (INT8) to:
- **Reduce model size** (INT8 uses 4x less memory than FP32)
- **Speed up inference** (INT8 operations are faster on hardware)
- **Lower power consumption** (smaller operations = less power)

## Why Do You Need a Calibration Dataset?

### The Problem

When converting from FP32/FP16 to INT8, you're essentially:
- Taking floating-point numbers (e.g., 0.123456789)
- Converting them to 8-bit integers (0-255 range)

But here's the challenge: **You need to know the range of values** your model will see during inference to do this conversion correctly.

### Example

Imagine your model has a weight value of `0.123456789` (FP32).

To convert to INT8, you need to:
1. Know the **min/max range** of all weights in that layer
2. Scale the value to fit in 0-255 range
3. Round to nearest integer

But if you don't know the range, you might:
- Use the wrong scale factor
- Lose important information
- Make the model inaccurate

### What is a Calibration Dataset?

A **calibration dataset** is a small, representative sample of your actual input data that you use to:

1. **Run inference** on the model with real inputs
2. **Observe the actual ranges** of values (activations, weights)
3. **Calculate optimal scale factors** for quantization
4. **Preserve accuracy** by using real data patterns

### Why Real Data Matters

Different inputs produce different activation patterns:
- A model trained on images might see values in range [-1.0, 1.0]
- But during inference, actual images might produce values in [-0.5, 0.8]
- If you quantize using the wrong range, you lose accuracy

### Typical Calibration Process

```python
# 1. Load your model
model = onnx.load("model.onnx")

# 2. Prepare calibration dataset (100-1000 samples is typical)
calibration_data = load_calibration_images()  # Your real input data

# 3. Run inference to observe value ranges
for image in calibration_data:
    output = model.run(image)
    # Quantization tool observes: min/max values, distributions, etc.

# 4. Calculate optimal quantization parameters
quantization_params = calculate_scale_factors(observed_ranges)

# 5. Apply quantization
quantized_model = quantize_model(model, quantization_params)
```

## For Your Use Case

### Current Implementation

Right now, `model_converter.py` does **configuration optimization** (selects best precision) but doesn't do actual quantization because:

1. **No calibration data**: We don't have representative input samples
2. **No quantization tools**: Would need ONNX Runtime quantization or TensorRT
3. **Simplified approach**: Just copies model and notes precision change

### What You'd Need to Add Real Quantization

```python
def _convert_precision_with_calibration(
    self,
    input_path: str,
    output_path: str,
    target_precision: str,
    calibration_data: List[np.ndarray]  # <-- This is what's missing
) -> str:
    """
    Convert model with actual quantization using calibration data.
    """
    if target_precision == "INT8":
        # Use ONNX Runtime quantization
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # This requires calibration data to calculate scale factors
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
            # calibration_data would be used here internally
        )
    
    return output_path
```

## Options for Getting Calibration Data

### Option 1: Use Your Video Data
```python
# You already have video files!
calibration_data = []
for frame in load_video_frames("data/samples/clip.mp4", num_frames=100):
    calibration_data.append(preprocess_frame(frame))
```

### Option 2: Use Your Profiling Data
```python
# You have profiling data with actual inputs
# Could extract representative frames from your CSV data
calibration_data = extract_frames_from_profiles(profiles_csv)
```

### Option 3: Generate Synthetic Data
```python
# Create synthetic data matching your model's expected input
calibration_data = [
    np.random.randn(1, 3, 640, 480).astype(np.float32)
    for _ in range(100)
]
```

## Current vs. Full Implementation

### What Works Now ✅
- **Configuration optimization**: Selects best precision (INT8/FP16/FP32)
- **Performance prediction**: Accurately predicts FPS/latency for each precision
- **Constraint handling**: Respects power/latency constraints
- **Model selection**: Chooses optimal configuration

### What's Simplified ⚠️
- **Model conversion**: Creates copy instead of actual quantization
- **No calibration**: Doesn't use real data for quantization
- **No actual INT8 conversion**: Would need quantization tools

### What You'd Add for Full Quantization

1. **Calibration data collection**:
   ```python
   def collect_calibration_data(video_path: str, num_samples: int = 100):
       # Extract frames from video
       # Preprocess them
       # Return list of numpy arrays
   ```

2. **Actual quantization**:
   ```python
   def quantize_model(model_path, calibration_data, output_path):
       # Use ONNX Runtime or TensorRT
       # Apply quantization with calibration data
       # Save quantized model
   ```

3. **Integration**:
   ```python
   # In model_converter.py
   calibration_data = collect_calibration_data(video_path)
   quantized_model = quantize_model(model_path, calibration_data, output_path)
   ```

## Summary

**Calibration dataset** = Small sample of your real input data used to:
- Observe actual value ranges during inference
- Calculate optimal quantization parameters
- Preserve model accuracy when converting to INT8

**Why it matters**: Without it, quantization might use wrong scale factors and hurt accuracy.

**For your project**: The optimization logic is complete. To add real quantization, you'd need:
1. Calibration data (can use your video files!)
2. Quantization tools (ONNX Runtime quantization or TensorRT)
3. Integration into the conversion pipeline

The framework is ready - just needs the actual quantization step added!

