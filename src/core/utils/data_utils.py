"""Utility functions for data loading, saving, and conversion."""

import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path


def convert_sample_data_to_edgetwin(
    csv_path: str,
    output_path: Optional[str] = None,
    sku_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Convert sample profiling CSV to EdgeTwin profiles_local.csv format.
    
    Expected input format (your sample data):
        timestamp, frame_id, engine_name, engine_precision, engine_batch,
        engine_shape, end_to_end_ms, power_mW, gpu_temp_C, ram_usage_MB, ...
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save converted CSV (if None, returns DataFrame only)
        sku_mapping: Optional dict mapping jetson_mode/platform to SKU names
        
    Returns:
        DataFrame in EdgeTwin format
    """
    df = pd.read_csv(csv_path)
    
    # Default SKU mapping
    if sku_mapping is None:
        sku_mapping = {
            "15W": "orin_nx",
            "25W": "orin_super",
            "10W": "orin_nano",
        }
    
    edgetwin_data = []
    
    for _, row in df.iterrows():
        # Parse engine_shape (assuming format like "[1, 3, 640, 640]")
        shape_str = row.get('engine_shape', '[1, 3, 640, 640]')
        if isinstance(shape_str, str):
            shape_str = shape_str.strip('[]')
            try:
                shape = [int(x.strip()) for x in shape_str.split(',')]
            except ValueError:
                shape = [1, 3, 640, 640]  # Default
        else:
            shape = shape_str if isinstance(shape_str, list) else [1, 3, 640, 640]
        
        # Extract resolution (assuming NCHW: [batch, channels, height, width])
        if len(shape) >= 4:
            height, width = int(shape[2]), int(shape[3])
        else:
            height, width = 640, 640
        
        # Determine SKU from jetson_mode
        jetson_mode = str(row.get('jetson_mode', '')).lower()
        sku = "orin_nx"  # Default
        for key, value in sku_mapping.items():
            if key.lower() in jetson_mode:
                sku = value
                break
        
        # Calculate FPS
        end_to_end_ms = float(row.get('end_to_end_ms', 0))
        fps = 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0
        
        edgetwin_data.append({
            'timestamp': row.get('timestamp'),
            'frame_id': row.get('frame_id', 0),
            'sku': sku,
            'model_name': row.get('engine_name', 'unknown'),
            'precision': row.get('engine_precision', 'FP16'),
            'resolution_h': height,
            'resolution_w': width,
            'batch_size': int(row.get('engine_batch', 1)),
            'frame_skip': 0,  # From experiment setup
            'latency_ms': end_to_end_ms,
            'trt_latency_ms': float(row.get('trt_latency_ms', 0)),
            'preprocess_ms': float(row.get('input_preprocess_ms', 0)),
            'postprocess_ms': float(row.get('postprocess_ms', 0)),
            'power_w': float(row.get('power_mW', 0)) / 1000.0,
            'power_mW': float(row.get('power_mW', 0)),
            'gpu_temp_C': float(row.get('gpu_temp_C', 0)),
            'gpu_util_percent': float(row.get('gpu_util_percent', 0)),
            'ram_usage_MB': float(row.get('ram_usage_MB', 0)),
            'gpu_mem_alloc_MB': float(row.get('gpu_mem_alloc_MB', 0)),
            'fps': fps,
            'jetson_mode': row.get('jetson_mode', ''),
        })
    
    edgetwin_df = pd.DataFrame(edgetwin_data)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edgetwin_df.to_csv(output_path, index=False)
    
    return edgetwin_df


def load_profile_results(csv_path: str, index: int = 0) -> Dict[str, Any]:
    """
    Load profile results from CSV and convert to EdgeTwin format.
    
    Args:
        csv_path: Path to profiles CSV
        index: Row index to load (default: 0)
        
    Returns:
        Profile results dict compatible with EdgeTwin
    """
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    
    return {
        "latency_ms": {
            "total": float(row.get('latency_ms', 0)),
            "preprocess": float(row.get('preprocess_ms', 0)),
            "inference": float(row.get('trt_latency_ms', 0)),
            "postprocess": float(row.get('postprocess_ms', 0)),
        },
        "fps": float(row.get('fps', 0)),
        "power_w": float(row.get('power_w', 0)),
        "memory_mb": float(row.get('ram_usage_MB', 0)),
        "sku": str(row.get('sku', 'orin_nx')),
        "iterations": 1,
    }

