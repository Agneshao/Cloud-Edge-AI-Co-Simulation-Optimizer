.PHONY: help setup install test lint format run-api run-web demo clean

help:
	@echo "EdgeTwin Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Set up development environment"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters"
	@echo "  format      - Format code"
	@echo "  run-api     - Run FastAPI server"
	@echo "  run-web     - Run Streamlit web UI"
	@echo "  demo        - Run demo workflow"
	@echo "  clean       - Clean artifacts"

setup:
	python -m venv venv
	. venv/bin/activate || . venv/Scripts/activate && pip install -e ".[dev]"

install:
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	black src/ tests/

run-api:
	uvicorn src.apps.api.server:app --reload --host 0.0.0.0 --port 8000

run-web:
	streamlit run src/apps/web/streamlit_app.py

demo:
	@echo "Running EdgeTwin demo..."
	python -m src.apps.cli.main full --model data/samples/yolov5n.onnx --video data/samples/clip.mp4

clean:
	rm -rf artifacts/reports/*
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

