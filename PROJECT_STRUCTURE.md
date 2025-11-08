# EdgeTwin Project Structure Documentation

**Last Updated**: 2025-11-06
**Current Branch**: dev_0
**Status**: Actively maintained locally (not pushed to GitHub)

---

## 📁 Project Hierarchy Overview

```
Cloud-Edge-AI-Co-Simulation-Optimizer/
├── src/                    # Core application code
├── configs/                # Configuration files
├── data/                   # Data, benchmarks, samples
├── docs/                   # Documentation
├── tests/                  # Test suite
├── artifacts/              # Build outputs (runtime)
├── Makefile                # Build automation
├── pyproject.toml          # Dependencies and package config
├── README.md               # Project overview
└── PROJECT_STRUCTURE.md    # This file (local documentation)
```

---

## 🔍 Top-Level Folders: Detailed Analysis

### 1. **`src/`** - Core Application Code
**Purpose**: Houses all production code organized by functional domain

#### Structure:
```
src/
├── core/                   # Business logic and algorithms
│   ├── profile/            # Hardware profiling
│   ├── predict/            # Performance prediction models
│   ├── optimize/           # Optimization algorithms
│   └── plan/               # Reporting and planning
├── apps/                   # User-facing interfaces
│   ├── api/                # FastAPI REST server
│   ├── cli/                # Command-line interface
│   └── web/                # Streamlit web UI
└── adapters/               # Hardware/cloud integration
    ├── jetson_adapter.py   # Jetson profiling
    └── isaac_sim_adapter.py # Isaac Sim integration
```

#### Module Roles:

**`src/core/`** - Core Business Logic
| Module | Files | Responsibility |
|--------|-------|-----------------|
| **profile** | `pipeline_profiler.py`, `stages.py` | ONNX model profiling on Jetson hardware (simulated/real) |
| **predict** | `power_models.py`, `latency_rule.py`, `thermal_rc.py`, `features.py`, `power_validation.py` | Performance prediction: power, latency, thermal |
| **optimize** | `knobs.py`, `search.py` | Configuration optimization using greedy/Bayesian search |
| **plan** | `reporter.py` | HTML report generation from profiling results |

**`src/apps/`** - User Interfaces
| Interface | Key Files | Purpose |
|-----------|-----------|---------|
| **api** | `server.py`, `schemas.py` | FastAPI REST endpoints (POST /profile, /optimize, etc.) |
| **cli** | `main.py` | Command-line: `python -m src.apps.cli.main profile ...` |
| **web** | `streamlit_app.py` | Streamlit UI for interactive profiling/optimization |

**`src/adapters/`** - Hardware Integration
| Adapter | Status | Role |
|---------|--------|------|
| **jetson_adapter.py** | 70% Complete | Jetson profiling (simulated works, real is stub) |
| **isaac_sim_adapter.py** | Template | Cloud simulation integration (mock responses) |

#### Architecture Review:
- ✅ **Strength**: Clear separation of concerns (profile → predict → optimize → report)
- ✅ **Strength**: Interfaces defined before implementation
- ⚠️ **Issue**: CLI/API endpoints not yet wired to core modules
- ⚠️ **Issue**: No error handling at app boundaries

---

### 2. **`configs/`** - Configuration Files
**Purpose**: Environment-agnostic configuration for all subsystems

#### Files:
```
configs/
├── defaults.yaml           # Global defaults (mode, paths, iterations)
├── constraints.yaml        # Performance constraints (FPS, power, thermal)
├── jetson_devices.yaml     # Jetson SKU specs (compute, memory, power modes)
└── optimization.yaml       # Search config (algorithm, knob bounds)
```

#### Key Content:

| File | Scope | Critical Settings |
|------|-------|-------------------|
| **defaults.yaml** | Application-wide | `mode`, `enable_*` flags, paths, log level |
| **constraints.yaml** | Performance limits | `min_fps: 30`, `max_power_w: 25`, `max_skin_temp_c: 70` |
| **jetson_devices.yaml** | Hardware specs | 5 SKUs (orin_super, orin_nx, orin_nano, xavier_nx, nano) with power modes |
| **optimization.yaml** | Search space | Algorithm (optuna/greedy), knob bounds (precision, resolution, batch size) |

#### Architecture Review:
- ✅ **Strength**: YAML-based, human-readable, easy to update
- ✅ **Strength**: Comprehensive device specs for 5 Jetson families
- ⚠️ **Issue**: No validation that configs are loaded correctly at startup
- 🔧 **Suggestion**: Add config schema validation (`pydantic` models)

---

### 3. **`data/`** - Data and Benchmarks
**Purpose**: Project datasets, benchmarks, and sample inputs

#### Structure:
```
data/
├── jetbenchdb/             # JetBench benchmark database
│   ├── boards.yaml         # Minimal board specs
│   ├── spec_curves.csv     # Scaling factors across SKU/precision/resolution
│   ├── profiles_local.csv  # Appended from local profiling runs
│   └── power_measurements_template.csv  # Template for empirical power data
└── samples/                # Sample inputs (EMPTY - needs files)
    ├── yolov5n.onnx        # ❌ Missing: tiny demo ONNX model
    └── clip.mp4            # ❌ Missing: 5-10s sample video
```

#### Data Roles:

| File | Purpose | Status |
|------|---------|--------|
| **boards.yaml** | Device reference data | ✅ Complete (5 devices) |
| **spec_curves.csv** | Performance scaling factors | ✅ Complete (9 configs) |
| **profiles_local.csv** | Empirical profiling results | ✅ Template ready |
| **power_measurements_template.csv** | Power calibration data | ✅ Template ready |
| **yolov5n.onnx** | Demo model | ❌ Missing |
| **clip.mp4** | Demo video | ❌ Missing |

#### Architecture Review:
- ✅ **Strength**: Clear separation: reference data vs. empirical data
- ✅ **Strength**: CSV format enables easy data collection and analysis
- 🔴 **Critical Issue**: `data/samples/` is empty - blocks demos and tests
- 🔧 **Suggestion**: Add `.gitkeep` or create placeholder files

---

### 4. **`tests/`** - Test Suite
**Purpose**: Unit and end-to-end tests

#### Structure:
```
tests/
├── unit/                   # Isolated component tests
│   ├── test_predict_monotonic.py  # ✅ Power model tests (27 lines)
│   └── __init__.py
├── e2e/                    # Full workflow tests
│   ├── test_full_flow.py   # ❌ Stub (18 lines, not implemented)
│   └── __init__.py
└── __init__.py
```

#### Coverage Analysis:

| Test File | Lines | Status | Coverage |
|-----------|-------|--------|----------|
| **test_predict_monotonic.py** | 27 | ✅ Complete | Power models monotonicity |
| **test_full_flow.py** | 18 | ❌ Stub | Full workflow (empty) |

#### Architecture Review:
- ⚠️ **Issue**: Only 1 real test file out of 2
- ⚠️ **Issue**: No tests for: model loading, error handling, API endpoints
- 🔧 **Suggestion**: Add test fixtures and coverage reports
- 🔧 **Suggestion**: Test: invalid configs, boundary conditions, API validation

---

### 5. **`docs/`** - Documentation
**Purpose**: Architecture, design decisions, and technical guides

#### Files:
```
docs/
├── ARCHITECTURE.md              # System design and component overview
├── POWER_MODEL_ANALYSIS.md      # Power modeling philosophy and roadmap
└── REPOSITORY_REVIEW.md         # Code quality review (248 lines)
```

#### Documentation Content:

| Doc | Focus | Quality |
|-----|-------|---------|
| **ARCHITECTURE.md** | System design, workflow, user story | ✅ Clear and concise |
| **POWER_MODEL_ANALYSIS.md** | Model philosophy, calibration roadmap | ✅ Thorough and honest |
| **REPOSITORY_REVIEW.md** | Code review, strengths, issues, recommendations | ✅ Comprehensive |

#### Architecture Review:
- ✅ **Strength**: Honest about limitations (e.g., estimated coefficients)
- ✅ **Strength**: Clear roadmap for power model improvements
- 🔧 **Suggestion**: Add API documentation (endpoints, schemas)
- 🔧 **Suggestion**: Add CLI quick start guide

---

### 6. **Root Level Files**

| File | Purpose | Status |
|------|---------|--------|
| **pyproject.toml** | Dependencies, package metadata, build config | ✅ Complete (68 lines) |
| **Makefile** | Build/run commands (setup, test, lint, etc.) | ✅ Complete (52 lines) |
| **README.md** | Project overview, quick start, structure | ✅ Complete (120 lines) |
| **PROJECT_STRUCTURE.md** | THIS FILE - structure and architecture docs | 🆕 Created |

---

## 🏗️ Overall Architecture

### Data Flow:
```
User Input (CLI/API/Web)
    ↓
[Apps Layer] → CLI | FastAPI | Streamlit
    ↓
[Core Layer] → Profile → Predict → Optimize → Report
    ↓
[Adapters] → Jetson Adapter | Isaac Sim Adapter
    ↓
[Configs] → defaults.yaml | constraints.yaml | jetson_devices.yaml
    ↓
[Data] → jetbenchdb/ | samples/
    ↓
Output (HTML Report | JSON Response | Console)
```

### Component Interactions:
1. **Profile Layer**: ONNX model → Pipeline stages → Metrics
2. **Predict Layer**: Metrics → Power/Latency/Thermal models → Predictions
3. **Optimize Layer**: Predictions + Constraints → Search algorithm → Optimal config
4. **Plan Layer**: Results → HTML report
5. **Adapters**: Hardware/Simulation abstraction for profiling

### Technology Stack:
- **Core**: Python 3.10+, ONNX Runtime
- **APIs**: FastAPI (REST), Streamlit (Web UI)
- **Config**: PyYAML
- **Optimization**: Optuna (future), Greedy (current)
- **Testing**: pytest
- **Tools**: Ruff, MyPy, Black

---

## 🎯 Optimization Suggestions

### 🔴 Critical (Blocking Issues)
1. **Missing sample data** (`data/samples/`)
   - Impact: Demos and tests cannot run
   - Fix: Create placeholder ONNX model and video files
   - Timeline: Immediate

2. **Unwired application endpoints**
   - Impact: CLI/API/Web don't actually run profiling
   - Files: `src/apps/cli/main.py`, `src/apps/api/server.py`
   - Fix: Connect endpoints to core modules
   - Timeline: High priority

### 🟡 High Priority
3. **Weak error handling**
   - Add try-catch in: `pipeline_profiler.py`, `search.py`
   - Add input validation at API boundaries
   - Timeline: Before first release

4. **Sparse test coverage**
   - Only 1 real test file
   - Missing: API tests, error cases, edge conditions
   - Timeline: Next iteration

5. **No logging configured**
   - Add logging setup to `src/__init__.py`
   - Configure: file + console handlers
   - Timeline: Before first release

### 🟢 Medium Priority
6. **Hardcoded power coefficients**
   - Add warnings in `power_models.py`
   - Reference: calibration guide in docs
   - Timeline: Next iteration

7. **API schemas need better validation**
   - Add Pydantic validators
   - Files: `src/apps/api/schemas.py`
   - Timeline: Next iteration

8. **Config validation missing**
   - Add schema validation at startup
   - Use: Pydantic `BaseSettings`
   - Timeline: Before production

### 🟢 Low Priority (Nice-to-Have)
9. **Add API documentation** (OpenAPI/Swagger)
10. **Add CLI help examples**
11. **Implement Optuna-based search** (optional)
12. **Add HTML report templates** (using Jinja2)

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Total Files | 46 |
| Python Modules | 34 |
| YAML Configs | 4 |
| Markdown Docs | 5 |
| Lines of Code | ~2,750 |
| Test Files | 2 |
| Test Lines | 45 |
| Config Lines | 230 |
| Doc Lines | 550+ |

---

## 🔄 Maintenance Schedule

This document is maintained **locally only** (not pushed to GitHub).

**Update triggers**:
- ✅ When files are added/deleted from top-level folders
- ✅ When significant architectural changes occur
- ✅ When module roles change
- ✅ When folder organization changes

**Last updated**: 2025-11-06 (Initial generation)

---

## 📝 Notes for Future Development

1. **Phase 1 (Current)**: Core modules complete, apps need wiring
2. **Phase 2 (Next)**: Wire CLI/API/Web, add error handling and tests
3. **Phase 3**: Real Jetson profiling and Isaac Sim integration
4. **Phase 4**: Calibrate power models with empirical data
5. **Phase 5**: Production hardening and optimization

---

**Document Maintainer**: Code Structure Review Assistant
**Next Review**: After next significant code changes
