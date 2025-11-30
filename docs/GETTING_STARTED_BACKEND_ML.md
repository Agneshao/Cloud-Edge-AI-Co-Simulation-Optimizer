# Getting Started: Backend & ML Development

This guide will help you get started working on the software backend and ML components of EdgeTwin.

## 🎯 Project Overview

EdgeTwin is a hardware-aware co-simulation platform that combines:
- **Profiling**: Measure model performance on Jetson hardware
- **Prediction**: ML models to predict latency, power, and thermal behavior
- **Optimization**: AI-driven search to find optimal configurations
- **Reporting**: Generate performance reports

## 📁 Key Components to Work On

### 1. **Prediction Models** (`src/core/predict/`)

**Current State:**
- ✅ Basic rule-based latency prediction (`latency_rule.py`)
- ✅ Simple power models (`power_models.py`, `power.py`)
- ✅ RC thermal model (`thermal_rc.py`)
- ✅ Feature engineering skeleton (`features.py`)
- ⚠️ Models use **estimated coefficients** - need calibration with real data

**What to Work On:**
- **Power Model Calibration**: Collect real Jetson power measurements and calibrate coefficients
- **Advanced ML Models**: Consider adding:
  - Gradient boosting (XGBoost, LightGBM) for power prediction
  - Neural networks for complex non-linear relationships
  - Ensemble methods combining multiple models
- **Feature Engineering**: Enhance `features.py` with:
  - Model architecture features (layer count, parameter count)
  - Hardware utilization features (GPU%, CPU%, memory%)
  - Temporal features (power history, thermal history)
- **Model Validation**: Use `power_validation.py` to compare model accuracy

**Files to Start With:**
- `src/core/predict/power_models.py` - Power prediction models
- `src/core/predict/power_validation.py` - Validation tools
- `src/core/predict/features.py` - Feature engineering

### 2. **Optimization Algorithms** (`src/core/optimize/`)

**Current State:**
- ✅ Basic greedy search (`search.py`)
- ✅ Configuration knobs defined (`knobs.py`)
- ⚠️ Greedy search can get stuck in local optima
- ⚠️ No Optuna/Bayesian optimization yet (despite config mentioning it)

**What to Work On:**
- **Implement Optuna Integration**: The config mentions Optuna but it's not implemented
  - Use Optuna for Bayesian optimization
  - Multi-objective optimization (Pareto front)
  - Pruning and early stopping
- **Improve Greedy Search**:
  - Add random restarts
  - Better resolution search (adaptive sampling)
  - Parallel evaluation
- **Objective Function Design**:
  - Multi-objective optimization (latency vs power vs accuracy)
  - Constraint handling (hard constraints vs penalties)
  - Weighted objective functions

**Files to Start With:**
- `src/core/optimize/search.py` - Add Optuna implementation
- `src/core/optimize/knobs.py` - Extend knob definitions if needed

### 3. **API Backend** (`src/apps/api/`)

**Current State:**
- ✅ FastAPI server structure (`server.py`)
- ✅ Pydantic schemas defined (`schemas.py`)
- ⚠️ Endpoints return **mock data** - need to wire up real implementations

**What to Work On:**
- **Wire Up Endpoints**: Replace mock data with real implementations
  - `/profile`: Connect to `PipelineProfiler`
  - `/predict`: Use prediction models
  - `/optimize`: Use optimization algorithms
- **Error Handling**: Add proper error handling and validation
- **Logging**: Add structured logging
- **Async Operations**: Make I/O operations async where appropriate
- **Testing**: Add API tests using `httpx` or `pytest`

**Files to Start With:**
- `src/apps/api/server.py` - Implement real endpoint logic
- `src/apps/api/schemas.py` - Extend schemas if needed

### 4. **Feature Engineering** (`src/core/predict/features.py`)

**Current State:**
- ✅ Basic feature extraction skeleton
- ⚠️ Very simple features - needs enhancement

**What to Work On:**
- Extract model architecture features from ONNX models
- Create interaction features (precision × resolution, etc.)
- Normalize/scale features appropriately
- Add feature importance analysis

## 🚀 Quick Start Guide

### 1. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import src.core.predict.power; print('✓ Installed')"
```

### 2. Run Tests

```bash
# Run all tests
make test
# or
pytest tests/

# Run specific test
pytest tests/unit/test_predict_monotonic.py -v
```

### 3. Start Development

#### Option A: Work on Prediction Models

```python
# Test power prediction
from src.core.predict.power import predict_power

power = predict_power(
    base_power_w=10.0,
    fps=30.0,
    precision="FP16",
    sku="orin_super",
    resolution=(640, 480)
)
print(f"Predicted power: {power}W")
```

#### Option B: Work on Optimization

```python
# Test optimization
from src.core.optimize.search import greedy_search
from src.core.optimize.knobs import ConfigKnobs

def objective(knobs):
    # Simple objective: minimize latency
    return 100.0  # Mock latency

best = greedy_search(objective)
print(f"Best config: {best}")
```

#### Option C: Work on API

```bash
# Start API server
make run-api
# or
uvicorn src.apps.api.server:app --reload

# Test endpoint
curl http://localhost:8000/health
```

## 📊 Data Collection & Model Training

### Collecting Power Data

1. **On Real Jetson Hardware:**
   ```bash
   # Use tegrastats to collect power measurements
   tegrastats --interval 1000 --logfile power_log.txt
   ```

2. **Parse and Store:**
   - Parse tegrastats logs
   - Store in `data/jetbenchdb/profiles_local.csv`
   - Include: timestamp, power_w, fps, precision, resolution, sku

3. **Calibrate Models:**
   ```python
   from src.core.predict.power_validation import calibrate_model_coefficients
   
   # Load your data
   data = load_power_measurements("data/jetbenchdb/profiles_local.csv")
   
   # Calibrate
   coefficients = calibrate_model_coefficients(data, model_type="SIMPLE_LINEAR")
   ```

### Training ML Models

If you want to add more sophisticated ML models:

1. **Prepare Dataset:**
   - Use `features.py` to extract features
   - Combine profile data + knob configurations
   - Split into train/validation/test

2. **Train Model:**
   ```python
   from sklearn.ensemble import GradientBoostingRegressor
   import pandas as pd
   
   # Load data
   df = pd.read_csv("data/jetbenchdb/profiles_local.csv")
   
   # Extract features
   X = build_features_from_dataframe(df)
   y = df["power_w"]
   
   # Train
   model = GradientBoostingRegressor()
   model.fit(X, y)
   
   # Save model
   import joblib
   joblib.dump(model, "models/power_model.pkl")
   ```

## 🔧 Development Workflow

### 1. Make Changes

- Edit files in `src/core/` for ML/backend logic
- Edit files in `src/apps/api/` for API endpoints
- Follow existing code style (see `pyproject.toml`)

### 2. Test Your Changes

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Lint code
make lint
# or
ruff check src/

# Format code
make format
# or
black src/
```

### 3. Validate Models

```python
# Compare model accuracy
from src.core.predict.power_validation import compare_models

results = compare_models(
    data=your_data,
    models=["SIMPLE_LINEAR", "POWER_MODE_AWARE"]
)
print(results)
```

## 📝 Recommended First Tasks

### Beginner Tasks

1. **Wire up API endpoints** (`src/apps/api/server.py`)
   - Replace mock data with real function calls
   - Add error handling
   - Test with curl or Postman

2. **Improve feature engineering** (`src/core/predict/features.py`)
   - Add more features (model size, layer count, etc.)
   - Add feature normalization
   - Test feature extraction

3. **Enhance greedy search** (`src/core/optimize/search.py`)
   - Add random restarts
   - Add early stopping
   - Add logging of search progress

### Intermediate Tasks

1. **Implement Optuna optimization** (`src/core/optimize/search.py`)
   - Add Optuna-based search function
   - Support multi-objective optimization
   - Add pruning strategies

2. **Calibrate power models** (`src/core/predict/power_models.py`)
   - Collect real power data
   - Fit coefficients using regression
   - Validate model accuracy

3. **Add advanced ML models**
   - Implement gradient boosting for power prediction
   - Add model persistence (save/load)
   - Add model versioning

### Advanced Tasks

1. **Multi-objective optimization**
   - Implement Pareto front optimization
   - Add constraint handling
   - Visualize optimization results

2. **Ensemble models**
   - Combine multiple prediction models
   - Weighted ensemble based on accuracy
   - Model selection based on context

3. **Real-time model updates**
   - Online learning from new measurements
   - Model retraining pipeline
   - A/B testing framework

## 🐛 Debugging Tips

1. **Check logs**: Add logging to track execution
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use IPython/Jupyter**: Interactive debugging
   ```python
   from IPython import embed; embed()
   ```

3. **Test in isolation**: Test functions independently
   ```python
   # Test power prediction
   from src.core.predict.power import predict_power
   result = predict_power(10.0, 30.0, "FP16", "orin_super", (640, 480))
   assert result > 0
   ```

## 📚 Resources

- **Optuna Docs**: https://optuna.org/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Jetson Power Management**: NVIDIA Jetson documentation
- **ONNX Runtime**: https://onnxruntime.ai/

## 🤝 Next Steps

1. Pick a component to work on (prediction, optimization, or API)
2. Read the existing code and TODOs
3. Make small, testable changes
4. Run tests to verify
5. Iterate and improve!

Good luck! 🚀

