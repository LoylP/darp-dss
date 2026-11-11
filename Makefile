.PHONY: install test test-all precompute api ui

install:
	python -m pip install -e ".[test]"

test:
	PYTHONPATH=. pytest -q -m "not slow"

test-all:
	PYTHONPATH=. pytest -q

precompute:
	python -m scripts.precompute --config configs/default.yaml --out artifacts/

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py
