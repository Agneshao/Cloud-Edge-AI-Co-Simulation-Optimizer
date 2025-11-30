"""
Example script to test the pipeline profiler implementation.

This demonstrates the new features:
- Video loading
- Power and memory measurement
- Integration with JetsonAdapter (when available)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.profile.pipeline_profiler import PipelineProfiler, profile_pipeline


def test_profiler_basic():
    """Test basic profiling without video."""
    print("=" * 60)
    print("Test 1: Basic Profiling (Dummy Data)")
    print("=" * 60)
    
    profiler = PipelineProfiler(sku="orin_super")
    
    # Note: This will use dummy data since we don't have a real model path
    # In real usage, provide path to ONNX model
    try:
        # This will fail if model doesn't exist, but shows the API
        results = profiler.profile(
            model_path="data/samples/yolov5n.onnx",  # Update with real path
            iterations=5
        )
        
        print(f"Latency (ms):")
        print(f"  Preprocess: {results['latency_ms']['preprocess']:.2f}")
        print(f"  Inference: {results['latency_ms']['inference']:.2f}")
        print(f"  Postprocess: {results['latency_ms']['postprocess']:.2f}")
        print(f"  Total: {results['latency_ms']['total']:.2f}")
        print(f"FPS: {results['fps']:.2f}")
        print(f"Power: {results['power_w']:.2f}W")
        print(f"Memory: {results['memory_mb']:.2f}MB")
        print(f"SKU: {results['sku']}")
        
    except FileNotFoundError as e:
        print(f"Model file not found (expected): {e}")
        print("This is normal - provide a real ONNX model path to test")
    except Exception as e:
        print(f"Error: {e}")


def test_profiler_with_video():
    """Test profiling with video input."""
    print("\n" + "=" * 60)
    print("Test 2: Profiling with Video")
    print("=" * 60)
    
    profiler = PipelineProfiler(sku="orin_super")
    
    try:
        # This will load actual video frames if video exists
        results = profiler.profile(
            model_path="data/samples/yolov5n.onnx",
            video_path="data/samples/clip.mp4",  # Update with real path
            iterations=5
        )
        
        print("✓ Video loaded successfully")
        print(f"FPS: {results['fps']:.2f}")
        print(f"Power: {results['power_w']:.2f}W")
        
    except FileNotFoundError as e:
        print(f"File not found (expected): {e}")
        print("This is normal - provide real model and video paths to test")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python")
    except Exception as e:
        print(f"Error: {e}")


def test_profiler_with_adapter():
    """Test profiling with JetsonAdapter."""
    print("\n" + "=" * 60)
    print("Test 3: Profiling with JetsonAdapter")
    print("=" * 60)
    
    try:
        from src.adapters.jetson_adapter import JetsonAdapter
        
        adapter = JetsonAdapter(sku="orin_super")
        profiler = PipelineProfiler(sku="orin_super", jetson_adapter=adapter)
        
        print(f"Real hardware detected: {adapter.is_real_hardware}")
        print(f"Power mode: {adapter.get_power_mode()}")
        
        # When JetsonAdapter._profile_real() is implemented,
        # this will use real power/memory measurements
        print("✓ JetsonAdapter integrated")
        print("  (Will use real measurements when on hardware)")
        
    except Exception as e:
        print(f"Error: {e}")


def test_statistics():
    """Test detailed statistics output."""
    print("\n" + "=" * 60)
    print("Test 4: Detailed Statistics")
    print("=" * 60)
    
    profiler = PipelineProfiler(sku="orin_super")
    
    try:
        results = profiler.profile(
            model_path="data/samples/yolov5n.onnx",
            iterations=10
        )
        
        if "latency_stats" in results:
            print("Latency Statistics (Total):")
            stats = results["latency_stats"]["total"]
            print(f"  Mean: {stats['mean']:.2f} ms")
            print(f"  Std: {stats['std']:.2f} ms")
            print(f"  Min: {stats['min']:.2f} ms")
            print(f"  Max: {stats['max']:.2f} ms")
        
    except FileNotFoundError:
        print("Model file not found (expected)")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Pipeline Profiler Implementation Tests")
    print("=" * 60 + "\n")
    
    test_profiler_basic()
    test_profiler_with_video()
    test_profiler_with_adapter()
    test_statistics()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Install dependencies: pip install opencv-python psutil")
    print("2. Provide real model and video paths")
    print("3. When hardware is available, JetsonAdapter will use real measurements")
    print("4. Check docs/PIPELINE_PROFILER_IMPLEMENTATION.md for details")


if __name__ == "__main__":
    main()

