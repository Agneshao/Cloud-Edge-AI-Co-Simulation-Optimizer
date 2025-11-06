"""Test monotonicity properties of prediction models."""

import pytest
from src.core.predict.latency_rule import predict_latency


def test_latency_precision_monotonic():
    """Test that INT8 < FP16 < FP32 for latency."""
    base_latency = 50.0
    
    int8_latency = predict_latency(base_latency, "orin_super", "INT8", (640, 480))
    fp16_latency = predict_latency(base_latency, "orin_super", "FP16", (640, 480))
    fp32_latency = predict_latency(base_latency, "orin_super", "FP32", (640, 480))
    
    assert int8_latency < fp16_latency < fp32_latency


def test_latency_resolution_monotonic():
    """Test that latency increases with resolution."""
    base_latency = 50.0
    
    low_res = predict_latency(base_latency, "orin_super", "FP16", (320, 240))
    mid_res = predict_latency(base_latency, "orin_super", "FP16", (640, 480))
    high_res = predict_latency(base_latency, "orin_super", "FP16", (1280, 960))
    
    assert low_res < mid_res < high_res

