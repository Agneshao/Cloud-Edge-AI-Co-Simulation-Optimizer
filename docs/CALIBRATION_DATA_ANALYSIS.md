# Calibration Data Analysis: Your CSV Data

## What Your CSV Data Contains

Your CSV row has:
- ✅ **Performance metrics**: `trt_latency_ms`, `power_mW`, `gpu_temp_C`, etc.
- ✅ **Configuration info**: `engine_precision`, `engine_batch`, `engine_shape`
- ✅ **Model info**: `engine_name`, `engine_size_MB`
- ✅ **Source info**: `frame_source_name: YOLOvideo`
- ❌ **NO actual input data**: No image/frame tensors

## Can You Use This for Calibration?

### Short Answer: **Not directly, but you can derive it!**

### What's Missing

For quantization calibration, you need:
- **Actual input tensors**: The image/frame data that was fed to the model
- **Not just metrics**: Performance numbers don't help with quantization

### What You Have

Your CSV tells you:
- What model was used (`yolov8n_fp16_static`)
- What input shape was used (`[1, 3, 640, 640]`)
- What source was used (`YOLOvideo`)
- What performance was achieved

### Solution: Use the Video Source

Since your CSV shows `frame_source_name: YOLOvideo`, you likely have:
- A video file that was processed
- The actual frames that were used

**You can extract calibration data from that video!**

## How to Get Calibration Data

### Option 1: Extract from Video File (Recommended)

```python
import cv2
import numpy as np
from pathlib import Path

def extract_calibration_frames(
    video_path: str,
    num_frames: int = 100,
    target_shape: tuple = (640, 640)
) -> list:
    """
    Extract frames from video for calibration.
    
    Args:
        video_path: Path to video file (e.g., "YOLOvideo.mp4")
        num_frames: Number of frames to extract (100-1000 is typical)
        target_shape: Target resolution (from your CSV: 640x640)
    
    Returns:
        List of preprocessed frames ready for model input
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)  # Sample evenly
    
    frame_idx = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            # Preprocess frame to match model input
            # 1. Resize to target shape
            frame_resized = cv2.resize(frame, target_shape)
            
            # 2. Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # 4. Convert HWC to CHW (channels first)
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            
            # 5. Add batch dimension: CHW -> NCHW
            frame_batched = np.expand_dims(frame_chw, axis=0)
            
            frames.append(frame_batched)
        
        frame_idx += 1
    
    cap.release()
    return frames

# Usage
calibration_data = extract_calibration_frames(
    video_path="data/samples/YOLOvideo.mp4",  # Your video file
    num_frames=100,
    target_shape=(640, 640)  # From your CSV: engine_shape
)
```

### Option 2: Use Your Existing Video Processing

If you already have a video file that was processed:
```python
# You might already have this in your pipeline
# Just extract frames from the same video that generated your CSV data
calibration_data = extract_calibration_frames("path/to/YOLOvideo.mp4")
```

### Option 3: Generate Synthetic Data (Fallback)

If you don't have the video file:
```python
def generate_synthetic_calibration(
    num_samples: int = 100,
    shape: tuple = (1, 3, 640, 640)
) -> list:
    """
    Generate synthetic data matching your model's input shape.
    
    Note: This is less ideal than real data, but works if video is unavailable.
    """
    return [
        np.random.rand(*shape).astype(np.float32)
        for _ in range(num_samples)
    ]

# Usage
calibration_data = generate_synthetic_calibration(
    num_samples=100,
    shape=(1, 3, 640, 640)  # From your CSV: engine_shape
)
```

## What You Need to Add

### 1. Video File Path

Your CSV shows `frame_source_name: YOLOvideo`, so you need:
- The actual video file (e.g., `YOLOvideo.mp4`, `clip.mp4`)
- Or access to where frames were stored

### 2. Frame Extraction Function

Add to `src/core/optimize/model_converter.py`:

```python
def collect_calibration_data(
    video_path: Optional[str] = None,
    num_frames: int = 100,
    model_shape: tuple = (1, 3, 640, 640)
) -> List[np.ndarray]:
    """
    Collect calibration data from video or generate synthetic.
    
    Args:
        video_path: Path to video file (if available)
        num_frames: Number of calibration frames
        model_shape: Model input shape (NCHW format)
    
    Returns:
        List of preprocessed frames
    """
    if video_path and Path(video_path).exists():
        # Extract from video (use function above)
        return extract_calibration_frames(video_path, num_frames)
    else:
        # Fallback to synthetic
        return generate_synthetic_calibration(num_frames, model_shape)
```

### 3. Update Model Converter

```python
def optimize_for_fps(
    self,
    model_path: str,
    target_fps: float,
    video_path: Optional[str] = None,  # <-- Add this
    output_path: Optional[str] = None,
    max_power_w: Optional[float] = None
) -> Dict[str, Any]:
    # ... existing code ...
    
    if target_precision == "INT8":
        # Collect calibration data
        calibration_data = collect_calibration_data(
            video_path=video_path,
            num_frames=100,
            model_shape=self._get_model_resolution(model_path)
        )
        
        # Use calibration data for quantization
        optimized_path = self._convert_precision_with_calibration(
            model_path, output_path, "INT8", calibration_data
        )
```

## Summary

### Your CSV Data: ✅ Useful for
- Understanding what was tested
- Baseline performance metrics
- Configuration information
- **But NOT for calibration** (no actual input tensors)

### What You Need: ✅
- **Video file** that was processed (e.g., `YOLOvideo.mp4`)
- **Frame extraction** function (I can add this)
- **Calibration data collection** (100-1000 frames)

### Next Steps

1. **Find your video file**: Look for `YOLOvideo.mp4` or similar
2. **Add frame extraction**: I can add this to `model_converter.py`
3. **Integrate calibration**: Use extracted frames for quantization

**Bottom line**: Your CSV is great for understanding performance, but you need the actual video file (or frames) for calibration. The CSV tells you what video was used - now you just need to extract frames from it!

Would you like me to add the calibration data collection function to your codebase?

