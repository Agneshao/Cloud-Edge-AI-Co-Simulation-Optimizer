# EdgeTwin

**Hardware-aware co-simulation platform for robotics AI**

EdgeTwin combines real Jetson profiling, cloud simulation (Isaac Sim), and AI-driven optimization to help teams verify performance before deploying to hardware.

## Features

- 🤖 **Jetson-Aware Profiling**: Simulated (and real) hardware profiling for NVIDIA Jetson families
- ☁️ **Cloud Simulation**: Isaac Sim integration for scenario-based testing
- 🎯 **AI-Driven Optimization**: Automated configuration tuning (precision, resolution, batch size, etc.)
- 📊 **Performance Reports**: HTML reports with profiling and optimization results

## Project Structure

```
edgetwin/
├─ src/
│  ├─ core/                              # Core logic modules
│  │  ├─ profile/                        # Jetson-aware profiling
│  │  │  ├─ stages.py                    # Pipeline stage shims
│  │  │  └─ pipeline_profiler.py         # Main profiling logic
│  │  ├─ predict/                         # Performance prediction models
│  │  │  ├─ features.py                  # Feature engineering
│  │  │  ├─ latency_rule.py               # Latency prediction
│  │  │  ├─ power.py                     # Power consumption models
│  │  │  └─ thermal_rc.py                # Thermal modeling
│  │  ├─ optimize/                       # Optimization algorithms
│  │  │  ├─ knobs.py                     # Configuration knobs
│  │  │  └─ search.py                    # AI-driven search
│  │  └─ plan/                           # Reporting
│  │     └─ reporter.py                  # HTML report generation
│  ├─ apps/                              # Application interfaces
│  │  ├─ api/                            # FastAPI REST server
│  │  ├─ cli/                            # Command-line interface
│  │  └─ web/                            # Streamlit web UI
│  └─ adapters/                          # Hardware/cloud adapters
│     ├─ jetson_adapter.py               # Jetson hardware integration
│     └─ isaac_sim_adapter.py            # Isaac Sim integration
├─ configs/
│  ├─ defaults.yaml                      # Default configuration
│  ├─ jetson_devices.yaml                # Jetson SKU specifications
│  ├─ constraints.yaml                   # Performance constraints
│  └─ optimization.yaml                  # Optimization parameters
├─ data/
│  ├─ jetbenchdb/
│  │  ├─ boards.yaml                    # minimal device specs
│  │  ├─ spec_curves.csv                # seed scaling factors across sku/precision/res
│  │  └─ profiles_local.csv             # appended profiles from local runs
│  └─ samples/
│     ├─ yolov5n.onnx                   # tiny demo model (placeholder ok)
│     └─ clip.mp4                       # 5–10s sample video
├─ artifacts/
│  └─ reports/                           # Generated HTML reports
├─ tests/
│  ├─ unit/                              # Unit tests
│  └─ e2e/                               # End-to-end tests
├─ docs/
│  └─ ARCHITECTURE.md                    # Architecture documentation
├─ Makefile                              # Build and run commands
├─ pyproject.toml                        # Dependencies and configuration
└─ README.md                             # This file
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or
pip install -e .
```

### Usage

#### API Server

```bash
make run-api
# or
uvicorn src.apps.api.server:app --reload
```

#### Web UI

```bash
make run-web
# or
streamlit run src/apps/web/streamlit_app.py
```

#### CLI

```bash
python -m src.apps.cli.main profile --model data/samples/model.onnx --sku orin_super
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format
```

## License

MIT
