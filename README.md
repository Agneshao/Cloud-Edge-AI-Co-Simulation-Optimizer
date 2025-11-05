### Project Structure:

edgematchpp/
├─ src/
│  ├─ edgematchpp_core/                 # PURE logic only
│  │  ├─ profile/
│  │  │  ├─ stages.py                   # tiny preprocess/infer/postprocess shims
│  │  │  └─ pipeline_profiler.py        # times the 3 stages on local machine
│  │  ├─ predict/
│  │  │  ├─ features.py                 # build features from profile + knobs
│  │  │  ├─ latency_rule.py             # rule-based latency scaling (sku × precision × res)
│  │  │  ├─ power.py                    # simple power model (base + k·fps)
│  │  │  └─ thermal_rc.py               # 1-pole RC thermal model (time-to-throttle)
│  │  ├─ optimize/
│  │  │  ├─ knobs.py                    # defines knobs: precision, res, frame_skip, batch
│  │  │  └─ search.py                   # greedy/Optuna search → best config
│  │  └─ plan/
│  │     └─ reporter.py                 # Jinja2 → HTML (and optional PDF) report
│  ├─ edgematchpp_apps/                 # I/O surfaces (thin)
│  │  ├─ api/
│  │  │  ├─ server.py                   # FastAPI: /profile /predict /optimize /plan
│  │  │  └─ schemas.py                  # Pydantic request/response models
│  │  ├─ cli/
│  │  │  └─ main.py                     # CLI that wires profile→predict→optimize→report
│  │  └─ web/
│  │     └─ streamlit_app.py            # simple UI: upload, sliders, run, view report
│  └─ edgematchpp_adapters/
│     └─ __init__.py                    # (placeholder; add onnx_runner/rose_cloud later)
├─ configs/
│  ├─ defaults.yaml                     # includes mode: hackathon, and global switches
│  ├─ devices.yaml                      # 3 SKUs (Orin Super + 2 extrapolated) with power modes
│  ├─ constraints.yaml                  # min_fps, max_power_w, max_skin_temp_c, ambient_c
│  └─ search.yaml                       # search budgets/bounds + weights for objective
├─ data/
│  ├─ jetbenchdb/
│  │  ├─ boards.yaml                    # minimal device specs
│  │  ├─ spec_curves.csv                # seed scaling factors across sku/precision/res
│  │  └─ profiles_local.csv             # appended profiles from local runs
│  └─ samples/
│     ├─ yolov5n.onnx                   # tiny demo model (placeholder ok)
│     └─ clip.mp4                       # 5–10s sample video
├─ artifacts/
│  └─ reports/                          # generated HTML reports
├─ tests/
│  ├─ unit/
│  │  └─ test_predict_monotonic.py      # res↑ ⇒ latency↑; INT8 < FP16 < FP32
│  └─ e2e/
│     └─ test_full_flow.py              # runs profile→predict→optimize→report smoke
├─ docs/
│  └─ ARCHITECTURE.md                   # 1-page diagram + user story (keep lean)
├─ Makefile                             # setup/run targets (demo, test, report)
├─ pyproject.toml                       # deps + ruff/mypy/pytest config
└─ README.md                            # quickstart + demo script
