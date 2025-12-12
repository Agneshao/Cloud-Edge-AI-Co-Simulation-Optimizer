# EdgeTwin MCP Tools Reference

**Endpoint:** `https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp`

---

## 1. `list_logs`
Lists all CSV log files in SmartBucket.

**Input:** None

**Response:**
```json
{
  "file_count": 4,
  "storage": "SmartBucket",
  "files": [
    {
      "filename": "inference_log_20251201_202925.csv",
      "size": 6641927,
      "uploaded": "2025-12-11T13:56:57.501Z"
    }
  ]
}
```

---

## 2. `load_log`
Loads a CSV file and returns detailed performance metrics.

**Input:**
```json
{ "filename": "inference_log_20251201_211136.csv" }
```

**Response:**
```json
{
  "file": "inference_log_20251201_211136.csv",
  "total_frames": 16279,
  "latency": {
    "avg_latency_ms": 28.606,
    "p50_latency_ms": 28.705,
    "p95_latency_ms": 31.721,
    "p99_latency_ms": 34.816,
    "max_latency_ms": 62.843,
    "min_latency_ms": 23.776,
    "jitter_std_ms": 2.16,
    "avg_trt_inference_ms": 12.499
  },
  "throughput": { "estimated_fps": 34.96 },
  "reliability": {
    "total_frame_drops": 76,
    "drop_rate": 0.0047,
    "avg_queue_delay_ms": 38.426,
    "max_queue_delay_ms": 72.298
  },
  "thermal": { "avg_temp_c": 48.66, "max_temp_c": 49.72, "min_temp_c": 45.03 },
  "power": { "avg_power_mw": 5862.84, "max_power_mw": 6730, "avg_power_w": 5.86 },
  "gpu": { "avg_util_percent": 35.8, "max_util_percent": 88 },
  "deployment_config": {
    "engine_name": "yolov8n_fp16_static",
    "engine_precision": "FP16",
    "engine_batch": 1,
    "engine_shape": "[1, 3, 640, 640]",
    "jetson_mode": "NV Power Mode: 15W",
    "platform": "Linux-5.15.148-tegra-aarch64-with-glibc2.35",
    "tensorrt_version": 10.3,
    "cuda_version": 12.2
  }
}
```

---

## 3. `recommend_config`
Analyzes log metrics and provides optimization recommendations.

**Input:**
```json
{ "filename": "inference_log_20251201_211136.csv" }
```

**Response:**
```json
{
  "filename": "inference_log_20251201_211136.csv",
  "current_config": {
    "engine_name": "yolov8n_fp16_static",
    "engine_precision": "FP16",
    "engine_batch": 1,
    "jetson_mode": "NV Power Mode: 15W"
  },
  "metrics_summary": {
    "avg_latency_ms": 28.606,
    "p95_latency_ms": 31.721,
    "estimated_fps": 34.96,
    "drop_rate": 0.0047,
    "avg_temp_c": 48.66,
    "avg_power_w": 5.86
  },
  "recommendations": [
    "LOW GPU UTILIZATION: GPU is underutilized. Consider increasing batch size to 2 or 4 for better throughput."
  ]
}
```

---

## 4. `predict_latency`
Uses Cerebras AI to predict performance for a proposed config.

**Input:**
```json
{
  "current_metrics": {
    "avg_latency_ms": 28.6,
    "avg_temp_c": 48.7,
    "gpu_util_percent": 35.8,
    "power_w": 5.86
  },
  "proposed_config": {
    "precision": "INT8",
    "batch_size": 2,
    "resolution": 640,
    "power_mode": "15W"
  }
}
```

**Response (standardized schema):**
```json
{
  "predicted_latency_ms": 18.5,
  "predicted_temp_c": 52.0,
  "predicted_power_w": 6.2,
  "risk_level": "low",
  "recommended_safe_mode": "fast",
  "explanation": "INT8 precision provides ~1.5x speedup over FP16. Batch size 2 slightly increases throughput. Temperature remains safe at 52C."
}
```

---

## 5. `evaluate_config`
Evaluates if a proposed config is safe to deploy (uses Cerebras AI).

**Input:**
```json
{
  "log_filename": "inference_log_20251201_211136.csv",
  "proposed_config": {
    "precision": "INT8",
    "batch_size": 4,
    "resolution": 416,
    "power_mode": "30W"
  },
  "constraints": {
    "max_latency_ms": 50,
    "max_temp_c": 75,
    "max_power_w": 25
  }
}
```

**Response:**
```json
{
  "prediction": {
    "predicted_latency_ms": 12.3,
    "predicted_temp_c": 58.0,
    "predicted_power_w": 12.5,
    "risk_level": "low",
    "recommended_safe_mode": "fast",
    "explanation": "INT8 at 416 resolution with batch 4 significantly reduces latency. 30W mode provides headroom."
  },
  "evaluation": {
    "safe_to_deploy": true,
    "violations": []
  },
  "baseline_metrics": {
    "avg_latency_ms": 28.606,
    "avg_temp_c": 48.66,
    "gpu_util_percent": 35.8,
    "power_w": 5.86
  },
  "proposed_config": {
    "precision": "INT8",
    "batch_size": 4,
    "resolution": 416,
    "power_mode": "30W"
  },
  "constraints_used": {
    "max_latency_ms": 50,
    "max_temp_c": 75,
    "max_power_w": 25
  }
}
```

---

## 6. `upload_log`
Upload a CSV log file to SmartBucket.

**Input:**
```json
{
  "filename": "new_log.csv",
  "csv_content": "timestamp,frame_id,engine_name,...\n1234567890,1,yolov8n,..."
}
```

**Response:**
```json
{
  "success": true,
  "filename": "new_log.csv",
  "size": 12345,
  "message": "File uploaded successfully to SmartBucket"
}
```

---

## 7. `delete_log`
Delete a log file from SmartBucket.

**Input:**
```json
{ "filename": "old_log.csv" }
```

**Response:**
```json
{
  "success": true,
  "filename": "old_log.csv",
  "message": "File deleted from SmartBucket"
}
```

---

## Shared Prediction Schema

Both `predict_latency` and `evaluate_config` return the same prediction format:

| Field | Type | Values |
|-------|------|--------|
| `predicted_latency_ms` | float | Predicted latency in ms |
| `predicted_temp_c` | float | Predicted GPU temp in Celsius |
| `predicted_power_w` | float | Predicted power in watts |
| `risk_level` | string | `"low"` \| `"medium"` \| `"high"` |
| `recommended_safe_mode` | string | `"fast"` \| `"safe"` |
| `explanation` | string | Natural language reasoning |

---

## Claude Desktop Config

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "edgetwin": {
      "url": "https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp"
    }
  }
}
```
