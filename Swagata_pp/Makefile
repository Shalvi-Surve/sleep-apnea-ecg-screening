.PHONY: install test lint fmt selftest clean

install:
	pip install -r requirements.txt
	pip install pytest ruff xgboost

test:
	pytest tests/ -v

lint:
	ruff check .

fmt:
	ruff check . --fix

# Full pipeline dry-run with synthetic data - no PhysioNet/network access needed.
selftest:
	python preprocess_apnea_ecg.py --self-test --out-dir ./processed_selftest
	python train_model.py \
		--train-csv ./processed_selftest/train_features_selftest.csv \
		--test-csv ./processed_selftest/test_features_selftest.csv \
		--out-dir ./model_output_selftest --cv-folds 4

clean:
	rm -rf processed_selftest model_output_selftest __pycache__ .pytest_cache */__pycache__
