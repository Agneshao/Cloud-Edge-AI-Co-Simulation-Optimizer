# 🗂️ EdgeTwin Project Structure - Quick Reference

**Last Updated**: 2025-11-06
**Maintained Locally Only** (not in GitHub)

---

## 📊 Folder Overview (One-Liner)

| Folder | Purpose | Status |
|--------|---------|--------|
| **`src/`** | 🔧 Core logic + 3 app interfaces + 2 adapters | 80% complete |
| **`configs/`** | ⚙️ YAML configs for devices, constraints, optimization | ✅ Complete |
| **`data/`** | 📦 Jetson benchmarks, sample inputs | ⚠️ Missing samples |
| **`docs/`** | 📚 Architecture, power models, code review | ✅ Complete |
| **`tests/`** | ✔️ Unit & E2E tests | 🟡 Sparse (1/2 done) |

---

## 🎯 Key Modules by Purpose

### Want to understand X? Go to:

**Performance Profiling** → `src/core/profile/pipeline_profiler.py`
- ONNX model loading and stage-by-stage timing

**Power Prediction** → `src/core/predict/power_models.py`
- 2 simple models (LINEAR, POWER_MODE_AWARE)
- See docs: `docs/POWER_MODEL_ANALYSIS.md`

**Optimization** → `src/core/optimize/search.py`
- Greedy search with multi-objective weighting

**Jetson Integration** → `src/adapters/jetson_adapter.py`
- Simulated working, real hardware is TODO

**REST API** → `src/apps/api/server.py`
- FastAPI endpoints (currently mock, need wiring)

**CLI** → `src/apps/cli/main.py`
- Command-line interface (currently stubs)

**Web UI** → `src/apps/web/streamlit_app.py`
- Streamlit dashboard (UI layout done, logic pending)

---

## 🚨 Critical Issues

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| Missing sample files | `data/samples/` | Demos won't run | 🔴 Immediate |
| Unwired endpoints | `src/apps/**` | Apps don't work | 🔴 High |
| No error handling | `src/core/**` | Silent failures | 🟡 High |
| Weak tests | `tests/` | No validation | 🟡 High |
| No logging | `src/` | Can't debug | 🟡 High |

---

## 📈 Dependency Flow

```
User Input
    ↓
[Apps: CLI/API/Web] → Request parsing
    ↓
[Core: Profile/Predict/Optimize/Plan] → Business logic
    ↓
[Adapters: Jetson/IsaacSim] → Hardware abstraction
    ↓
[Configs: YAML] → Configuration
    ↓
[Data: JetBench] → Reference data
    ↓
Output: Report/Response/UI
```

---

## 🔍 File Type Distribution

```
Python Modules:     34 files (~2,750 LOC)
Configurations:     4 files (230 LOC)
Documentation:      5 files (550+ LOC)
Tests:              2 files (45 LOC)
Build Config:       2 files (Makefile, pyproject.toml)
```

---

## 🛠️ Common Tasks

### "I want to add a new Jetson device"
→ Edit `configs/jetson_devices.yaml` and `data/jetbenchdb/boards.yaml`

### "I want to add a new power model"
→ Add to `src/core/predict/power_models.py` and `src/core/predict/power.py`

### "I want to understand the architecture"
→ Read `docs/ARCHITECTURE.md` (5 min) then `PROJECT_STRUCTURE.md` (10 min)

### "I want to fix the API"
→ Check `src/apps/api/server.py` → look for `# TODO` comments

### "I want to run tests"
→ Use `make test` (currently only tests power model monotonicity)

### "I want to add validation"
→ Add Pydantic models to `src/apps/api/schemas.py`

---

## 📂 Directory Tree (Compact)

```
Cloud-Edge-AI-Co-Simulation-Optimizer/
│
├── src/
│   ├── core/                          # Business logic
│   │   ├── profile/                   # ONNX profiling
│   │   ├── predict/                   # Power/latency/thermal models
│   │   ├── optimize/                  # Search algorithms
│   │   └── plan/                      # Report generation
│   │
│   ├── apps/                          # User interfaces
│   │   ├── api/                       # FastAPI REST
│   │   ├── cli/                       # Command-line
│   │   └── web/                       # Streamlit UI
│   │
│   └── adapters/                      # Hardware integration
│       ├── jetson_adapter.py
│       └── isaac_sim_adapter.py
│
├── configs/                           # Configuration
│   ├── defaults.yaml
│   ├── constraints.yaml
│   ├── jetson_devices.yaml
│   └── optimization.yaml
│
├── data/
│   ├── jetbenchdb/                    # Reference data
│   │   ├── boards.yaml
│   │   ├── spec_curves.csv
│   │   └── ...
│   └── samples/                       # ❌ EMPTY - needs files
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md
│   ├── POWER_MODEL_ANALYSIS.md
│   └── REPOSITORY_REVIEW.md
│
├── tests/
│   ├── unit/
│   │   └── test_predict_monotonic.py  # ✅ Works
│   └── e2e/
│       └── test_full_flow.py          # ❌ Stub
│
├── Makefile                           # Build automation
├── pyproject.toml                     # Dependencies
├── README.md                          # Overview
└── PROJECT_STRUCTURE.md               # Full documentation
```

---

## 🔄 How This Document is Maintained

This quick reference is automatically updated when:
- ✅ Files are added/removed from top-level folders
- ✅ Module purposes change
- ✅ Critical issues are resolved
- ✅ Architecture changes

**Status**: Actively maintained locally (not synced to GitHub)

---

## 💡 Quick Navigation

- **Want architecture details?** → `PROJECT_STRUCTURE.md`
- **Want to understand code?** → `docs/ARCHITECTURE.md`
- **Want to understand power models?** → `docs/POWER_MODEL_ANALYSIS.md`
- **Want code review feedback?** → `docs/REPOSITORY_REVIEW.md`
- **Want to run commands?** → `Makefile`
- **Want to fix issues?** → See "Critical Issues" section above

---

**Maintainer**: Code Structure Review Assistant
**Next Update**: After next significant changes
