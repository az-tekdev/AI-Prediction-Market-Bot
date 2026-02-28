.PHONY: help install test lint format clean run-backtest run-train run-trade docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make run-backtest  - Run backtest example"
	@echo "  make run-train     - Train model"
	@echo "  make run-trade     - Run live trading (dry-run)"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

lint:
	flake8 src tests main.py
	pylint src tests main.py

format:
	black src tests main.py
	isort src tests main.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.pyc */*.pyc */*/*.pyc
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf build dist *.egg-info

run-backtest:
	python main.py backtest --market-id test_market --model-type logistic --plot

run-train:
	python main.py train --model-type logistic

run-trade:
	python main.py trade --model-type logistic

docker-build:
	docker build -t ai-prediction-bot .

docker-run:
	docker run --env-file .env ai-prediction-bot
