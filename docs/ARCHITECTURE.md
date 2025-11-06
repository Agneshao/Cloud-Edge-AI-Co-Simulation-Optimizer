# EdgeTwin Architecture

## Overview

EdgeTwin is a hardware-aware co-simulation platform for robotics AI that combines:
1. **Real Jetson Profiling** - Hardware-aware performance measurement
2. **Cloud Simulation** - Isaac Sim integration for scenario testing
3. **AI-Driven Optimization** - Automated configuration tuning

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EdgeTwin Platform                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Profile    │───▶│   Predict    │───▶│  Optimize    │  │
│  │  (Jetson)    │    │  (Models)    │    │  (Search)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         └───────────────────┼────────────────────┘          │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │     Report      │                       │
│                    │   (HTML/PDF)    │                       │
│                    └─────────────────┘                       │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Isaac Sim Integration                     │  │
│  │         (Cloud Simulation for Scenarios)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Modules

1. **Profile** (`src/core/profile/`)
   - `stages.py`: Pipeline stage shims
   - `pipeline_profiler.py`: Main profiling logic
   - Jetson-aware metrics collection

2. **Predict** (`src/core/predict/`)
   - `latency_rule.py`: Rule-based latency prediction
   - `power.py`: Power consumption models
   - `thermal_rc.py`: Thermal modeling

3. **Optimize** (`src/core/optimize/`)
   - `knobs.py`: Configuration knobs
   - `search.py`: AI-driven search algorithms

4. **Plan** (`src/core/plan/`)
   - `reporter.py`: Report generation

### Application Layer

- **API** (`src/apps/api/`): FastAPI REST interface
- **CLI** (`src/apps/cli/`): Command-line interface
- **Web** (`src/apps/web/`): Streamlit UI

### Adapters

- **Jetson Adapter**: Hardware integration
- **Isaac Sim Adapter**: Cloud simulation integration

## Workflow

1. **Profile**: Measure model performance on Jetson hardware (or simulated)
2. **Predict**: Predict performance for different configurations
3. **Optimize**: Find optimal configuration using AI-driven search
4. **Simulate**: Run co-simulation in Isaac Sim
5. **Report**: Generate comprehensive performance reports

## User Story

A robotics engineer wants to deploy a perception model to Jetson Orin Super. They:
1. Upload their ONNX model to EdgeTwin
2. Run profiling to get baseline metrics
3. Use EdgeTwin's prediction models to explore configurations
4. Run optimization to find the best precision/resolution/batch settings
5. Verify the optimized config in Isaac Sim scenarios
6. Generate a report showing performance vs. constraints
7. Deploy with confidence knowing the model will meet requirements

