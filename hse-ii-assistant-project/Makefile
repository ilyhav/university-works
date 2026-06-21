.PHONY: help install data app eval drift test lint clean

help:
	@echo "make install  — установить зависимости (uv sync, либо pip -e .[dev])"
	@echo "make data     — сгенерировать учебный каталог вкладов и golden-набор"
	@echo "make app      — запустить Streamlit веб-форму (http://localhost:8501)"
	@echo "make eval     — прогнать валидацию качества на golden-наборе → отчёт"
	@echo "make drift    — симуляция дрейфа (сдвиг ключевой ставки) + алерты"
	@echo "make test     — pytest (ядро, метрики, guardrails)"
	@echo "make lint     — ruff check"

install:
	uv sync --extra dev || pip install -e ".[dev]"

data:
	python -m scripts.generate_dataset

app:
	streamlit run app/streamlit_app.py

eval:
	python -m scripts.run_eval

drift:
	python -m scripts.simulate_drift

test:
	pytest

lint:
	ruff check src app scripts tests
