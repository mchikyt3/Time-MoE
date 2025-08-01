# Time-MoE Tests

This directory contains comprehensive test scripts for Time-MoE functionality.

## Test Files

### Core Component Tests

- **`test_configuration.py`**: Unit tests for TimeMoeConfig
  - Tests default and custom configuration values
  - Tests configuration validation and constraints
  - Tests configuration serialization
  - Tests parameter conversion (e.g., horizon_lengths)

- **`test_models.py`**: Unit tests for Time-MoE model components
  - Tests individual components (embedding, attention, MLP, etc.)
  - Tests complete models (prediction, classification)
  - Tests utility functions (load balancing, repeat_kv)
  - Tests training/evaluation mode switching

- **`test_datasets.py`**: Unit tests for dataset classes
  - Tests GeneralDataset with various file formats (JSON, JSONL, NPY)
  - Tests BinaryDataset functionality
  - Tests file reading utilities
  - Tests data validation and loading

- **`test_utils.py`**: Unit tests for utility modules
  - Tests logging utilities (get_logger, log_in_local_rank_0)
  - Tests distributed utilities (get_world_size)
  - Tests environment variable handling

- **`test_runner.py`**: Unit tests for TimeMoeRunner
  - Tests model loading from scratch
  - Tests different task types (forecasting, classification)
  - Tests attention implementation selection
  - Tests batch size calculation logic

### Integration and End-to-End Tests

- **`test_integration.py`**: Integration tests for complete workflows
  - Tests end-to-end forecasting workflow
  - Tests sequence and token classification workflows
  - Tests multi-horizon forecasting
  - Tests gradient computation
  - Tests config serialization roundtrip

### Legacy Tests

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

- **`test_run_eval.py`**: Tests for evaluation functionality (currently empty)

## Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test files
cd tests
python test_configuration.py
python test_models.py
python test_datasets.py
python test_utils.py
python test_runner.py
python test_integration.py

# Run legacy tests
python test_freeze_backbone.py
python test_classification.py

# Or from project root
python tests/test_configuration.py
python tests/test_models.py
# ... etc
```

## Test Coverage

The test suite covers:

### Models and Configuration
- ✅ TimeMoeConfig validation and serialization
- ✅ Individual model components (embeddings, attention, MLP, etc.)
- ✅ Complete models (TimeMoeForPrediction, TimeMoeForSequenceClassification, TimeMoeForTokenClassification)
- ✅ Model training/evaluation modes
- ✅ Gradient computation
- ✅ Parameter counting and freezing

### Datasets
- ✅ GeneralDataset with multiple file formats
- ✅ BinaryDataset functionality
- ✅ File reading utilities
- ✅ Data validation

### Utilities
- ✅ Logging utilities
- ✅ Distributed training utilities
- ✅ Environment variable handling

### Runner and Integration
- ✅ TimeMoeRunner model loading
- ✅ End-to-end workflows
- ✅ Multi-task support (forecasting, classification)
- ✅ Configuration management

## Test Requirements

The tests use only built-in Time-MoE dependencies and require:
- torch
- transformers
- time_moe models, configurations, and utilities
- numpy (for dataset tests)

No additional testing frameworks are required beyond Python's built-in unittest.
