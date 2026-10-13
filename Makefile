# Makefile for Delhi-NCR PM2.5 Forecasting Project

.PHONY: init build test train clean help

INIT:
	python scripts/pull_cpcb.py
	python scripts/pull_openmeteo.py

build:
	python scripts/build_dataset.py

clean:
	rm -rf data/processed/*.npz
	rm -rf checkpoints/*.pt
	rm -rf logs/*.txt

help:
	@echo "Available targets:"
	@echo "  init    - Initialize data sources"
	@echo "  build   - Build dataset"
	@echo "  train   - Train model (use --config for specific model)"
	@echo "  clean   - Clean generated files"

train:
	python scripts/run_experiment.py --config $(CONFIG)

test:
	python -m pytest tests/
