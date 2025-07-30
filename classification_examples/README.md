# Time-MoE Classification Examples

This folder contains example scripts and utilities for Time-MoE classification fine-tuning.

## Files Overview

### 📊 Data Preparation
- **`create_sample_classification_data.py`**: Generate synthetic classification datasets for testing
  - Creates both sequence and token classification sample data in JSONL format
  - Useful for testing and understanding the data format requirements

- **`prepare_classification_data.py`**: Convert your data to Time-MoE classification format
  - Handles CSV to JSONL conversion with proper labeling
  - Supports both sequence and token classification formats
  - Includes data validation and preprocessing

### 🎯 Training Example
- **`classification_example.py`**: Complete end-to-end classification training example
  - Demonstrates both sequence and token classification workflows
  - Shows how to use different training configurations
  - Includes model evaluation and saving

### 🔮 Inference
- **`inference_classification.py`**: Load trained models and make predictions
  - Load saved classification models
  - Perform inference on new time series data
  - Export predictions in various formats

## Quick Start

1. **Generate sample data:**
   ```bash
   python create_sample_classification_data.py
   ```

2. **Run classification training:**
   ```bash
   python classification_example.py
   ```

3. **Use your own data:**
   ```bash
   python prepare_classification_data.py --input your_data.csv --output formatted_data.jsonl --task_type sequence_classification
   ```

4. **Perform inference:**
   ```bash
   python inference_classification.py --model_path ./trained_model --data_path test_data.jsonl
   ```

## Data Format

All classification scripts expect JSONL format:

**Sequence Classification (one label per series):**
```json
{"values": [1.2, 1.5, 1.8, ...], "label": 0}
{"values": [2.1, 2.3, 2.0, ...], "label": 1}
```

**Token Classification (one label per timestep):**
```json
{"values": [1.2, 1.5, 1.8, ...], "labels": [0, 0, 1, ...]}
{"values": [2.1, 2.3, 2.0, ...], "labels": [1, 1, 0, ...]}
```

For more details, see the main [Time-MoE README](../README.md#classification-fine-tuning).
