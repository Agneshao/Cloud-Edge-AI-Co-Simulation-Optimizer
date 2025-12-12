# Cloud-Edge AI Co-Simulation Optimizer

An intelligent edge computing optimization platform that uses AI to analyze, predict, and optimize Jetson edge device performance. Built with Raindrop Framework and powered by Cerebras AI.

## Overview

This project provides a comprehensive solution for managing and optimizing AI inference workloads on NVIDIA Jetson devices. It collects performance metrics, analyzes deployment configurations, and uses AI to predict optimal settings for balancing latency, throughput, power consumption, and thermal performance.

### Key Features

- **Performance Monitoring**: Track real-time metrics including latency, FPS, power consumption, temperature, and GPU utilization
- **AI-Powered Predictions**: Use Cerebras AI to predict performance outcomes for different configurations
- **Configuration Optimization**: Get intelligent recommendations for model precision, batch size, resolution, and power mode
- **Safety Validation**: Evaluate proposed configurations against constraints before deployment
- **MCP Integration**: Seamless integration with Claude Desktop via Model Context Protocol
- **Cloud Storage**: Automated log storage and management using SmartBucket

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Jetson Edge    │  logs   │  Raindrop Cloud  │   MCP   │ Claude Desktop  │
│  Device         ├────────>│  Platform        │<────────┤                 │
│  (YOLOv8)       │         │  + Cerebras AI   │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+ (for local MCP server)
- Raindrop CLI: `npm install -g @liquidmetal-ai/raindrop`
- Authenticated with Raindrop: `raindrop auth login`

### Installation

```bash
# Clone the repository
git clone https://github.com/Agneshao/Cloud-Edge-AI-Co-Simulation-Optimizer.git
cd Cloud-Edge-AI-Co-Simulation-Optimizer

# Install dependencies
npm install

# Install Python dependencies for local MCP server
pip install -r mcp_servers/jetson_logs_server/requirements.txt
```

### Deployment

```bash
# Generate TypeScript types
raindrop build generate

# Deploy to Raindrop Platform
npm run start
```

## MCP Tools Reference

The platform provides 7 MCP tools accessible via Claude Desktop:

### 1. `list_logs`
List all CSV log files stored in SmartBucket.

**Response:**
```json
{
  "file_count": 4,
  "files": [
    {
      "filename": "inference_log_20251201_211136.csv",
      "size": 6641927,
      "uploaded": "2025-12-11T13:56:57.501Z"
    }
  ]
}
```

### 2. `load_log`
Load and analyze a CSV log file to get detailed performance metrics.

**Input:**
```json
{ "filename": "inference_log_20251201_211136.csv" }
```

**Returns:** Comprehensive metrics including:
- Latency statistics (avg, p50, p95, p99, max, min, jitter)
- Throughput (FPS)
- Reliability (frame drops, queue delay)
- Thermal metrics (temperature)
- Power consumption
- GPU utilization
- Deployment configuration

### 3. `recommend_config`
Analyze metrics and provide optimization recommendations.

**Input:**
```json
{ "filename": "inference_log_20251201_211136.csv" }
```

**Returns:** Current config, metrics summary, and actionable recommendations.

### 4. `predict_latency`
Use Cerebras AI to predict performance for a proposed configuration.

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

**Returns:** Predicted latency, temperature, power, risk level, and AI explanation.

### 5. `evaluate_config`
Evaluate if a proposed configuration is safe to deploy.

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

**Returns:** Safety evaluation, constraint violations, and AI prediction.

### 6. `upload_log`
Upload a new CSV log file to SmartBucket.

### 7. `delete_log`
Delete a log file from SmartBucket.

## Claude Desktop Integration

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "edgetwin": {
      "url": "https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp"
    }
  }
}
```

## Project Structure

```
.
├── src/
│   ├── _app/                    # App-level configuration
│   │   ├── auth.ts             # JWT authentication
│   │   └── cors.ts             # CORS settings
│   └── jetson-logs/            # Main service handlers
│       ├── index.ts            # HTTP service endpoints
│       ├── index.test.ts       # Unit tests
│       └── raindrop.gen.ts     # Generated types
├── mcp_servers/
│   └── jetson_logs_server/     # Python MCP server
│       ├── server.py           # MCP server implementation
│       └── requirements.txt    # Python dependencies
├── data/
│   └── sample_logs/            # Sample inference logs
├── dist/                       # Compiled JavaScript
├── raindrop.manifest           # Raindrop app configuration
├── package.json
├── tsconfig.json
└── README.md
```

## Development Workflow

### 1. Local Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Lint code
npm run lint

# Format code
npm run format
```

### 2. Deploying Updates

```bash
# Build TypeScript
npm run build

# Validate configuration
raindrop build validate

# Deploy and start
npm run start
```

### 3. Monitoring

```bash
# Check deployment status
raindrop build status

# View real-time logs
raindrop logs tail

# Query historical logs
raindrop logs query --since 1h
```

## Sample Log Format

The system expects CSV logs with the following columns:

```csv
timestamp,frame_id,engine_name,engine_precision,batch_size,input_shape,
latency_ms,trt_inference_ms,preprocess_ms,postprocess_ms,
frame_dropped,queue_delay_ms,fps,gpu_temp_c,gpu_util_percent,
power_mw,jetson_mode,platform,tensorrt_version,cuda_version
```

## Configuration Optimization Strategies

The AI recommends optimizations based on:

1. **GPU Utilization**: Increase batch size if GPU is underutilized
2. **Thermal Headroom**: Suggest INT8 precision or higher power modes if temperature is safe
3. **Latency**: Recommend lower resolution or smaller batch sizes for real-time requirements
4. **Power Efficiency**: Balance performance with power constraints
5. **Frame Drops**: Identify queue bottlenecks and recommend fixes

## Technology Stack

- **Framework**: [Raindrop Framework](https://liquidmetal.ai) - Serverless edge computing platform
- **AI Model**: Cerebras AI (via Raindrop AI integration)
- **HTTP Server**: Hono.js - Lightweight web framework
- **Storage**: SmartBucket - AI-powered object storage
- **Protocol**: MCP (Model Context Protocol)
- **Runtime**: Cloudflare Workers

## Use Cases

### 1. Real-Time Performance Tuning
Monitor live inference performance and get instant optimization suggestions.

### 2. Pre-Deployment Validation
Test configuration changes in the cloud before deploying to edge devices.

### 3. Multi-Device Management
Track and optimize performance across a fleet of Jetson devices.

### 4. Energy Optimization
Find the optimal balance between performance and power consumption.

### 5. Constraint Validation
Ensure configurations meet latency, temperature, and power requirements.

## Environment Variables

Set in `raindrop.manifest` or via Raindrop CLI:

```raindrop
application "jetson-optimizer" {
  env "CEREBRAS_API_KEY" {
    secret = true
  }
}
```

## Supported Jetson Configurations

- **Platforms**: Jetson Orin Nano, Jetson Orin NX, Jetson AGX Orin
- **Power Modes**: 7W, 15W, 25W, 30W, 50W (device-dependent)
- **Precisions**: FP32, FP16, INT8
- **Batch Sizes**: 1, 2, 4, 8
- **Resolutions**: 416x416, 640x640, 1280x1280

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software. All rights reserved.

## Acknowledgments

- Built with [Raindrop Framework](https://liquidmetal.ai)
- Powered by Cerebras AI
- Designed for NVIDIA Jetson platforms

## Contact

For questions or support, please open an issue on GitHub.

## Roadmap

- [ ] Multi-model support (beyond YOLOv8)
- [ ] Automated A/B testing for configurations
- [ ] Real-time streaming metrics
- [ ] Advanced anomaly detection
- [ ] Cost optimization recommendations
- [ ] Integration with MLOps platforms

---

**Note**: This is an active research project. Performance predictions are estimates and should be validated in your specific deployment environment.
