# Time-MoE Tests

This directory contains test scripts for Time-MoE functionality.

## Test Files

- **`test_freeze_backbone.py`**: Comprehensive tests for freeze_backbone functionality
  - Tests freeze_backbone for forecasting models
  - Tests freeze_backbone for classification models
  - Validates parameter counting and freezing behavior
  - Ensures backbone contains majority of parameters

- **`test_classification.py`**: Unit tests for classification functionality
  - Tests sequence and token classification model creation
  - Tests freeze_backbone with classification models
  - Tests forward passes and output shapes
  - Tests different pooling strategies
  - Tests configuration validation
  - Tests data format validation

- **`test_run_eval.py`**: Unit tests for evaluation functionality in `run_eval.py`
  - Tests all metric classes (MSE, MAE, Accuracy, F1)
  - Tests TimeMoE model initialization for all task types
  - Tests model prediction methods for forecasting and classification
  - Tests the main evaluate function with mocked components
  - Tests distributed evaluation setup
  - Tests argument parsing and validation logic
  - Tests helper functions like `count_num_tensor_elements` and `setup_nccl`

## Running Tests

```bash
# Run all freeze_backbone tests
cd tests
python test_freeze_backbone.py

# Run all classification tests
python test_classification.py

# Run all evaluation tests
python test_run_eval.py

# Or from project root
python tests/test_freeze_backbone.py
python tests/test_classification.py
python tests/test_run_eval.py

# Run all tests with unittest
python -m unittest discover tests/

# Run with verbose output
python -m unittest discover tests/ -v
```

## Test Requirements

The tests use only built-in Time-MoE dependencies and require:
- torch
- time_moe models and configurations

No additional testing frameworks are required.
