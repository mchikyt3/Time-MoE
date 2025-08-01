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

### 🔧 Advanced Features
- **`pooling_strategies_example.py`**: Examples of different pooling strategies for sequence classification
  - Demonstrates various pooling methods (last_token, mean, max, attention, etc.)
  - Shows how different strategies affect classification performance

### 🔮 Evaluation
- Use the main **`run_eval.py`** script (in root directory) for evaluating classification models
  - Supports both sequence and token classification evaluation
  - Provides accuracy and F1 score metrics
  - Unified evaluation framework for both forecasting and classification tasks

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

4. **Evaluate classification models:**
   ```bash
   # Sequence classification
   python ../run_eval.py --model path/to/model --data data.jsonl --task_type sequence_classification --num_classes 5
   
   # Token classification  
   python ../run_eval.py --model path/to/model --data data.jsonl --task_type token_classification --num_classes 3
   ```

## Data Format

All classification scripts expect JSONL format:

**Sequence Classification (one label per series):**
```json
{"sequence": [1.2, 1.5, 1.8, ...], "label": 0}
{"sequence": [2.1, 2.3, 2.0, ...], "label": 1}
```

**Token Classification (one label per timestep):**
```json
{"sequence": [1.2, 1.5, 1.8, ...], "label": [0, 0, 1, ...]}
{"sequence": [2.1, 2.3, 2.0, ...], "label": [1, 1, 0, ...]}
```

## Freeze Backbone Training

The `--freeze_backbone` flag is available for all task types to reduce training time and memory usage:

```bash
# Classification with frozen backbone (recommended for small datasets)
python ../main.py -d data.jsonl --task_type sequence_classification --num_classes 3 --freeze_backbone

# Forecasting with frozen backbone (for domain adaptation)
python ../main.py -d data.jsonl --task_type forecasting --freeze_backbone --learning_rate 1e-3
```

When enabled, only the task-specific heads are trained while the backbone remains frozen, typically training <10% of total parameters.

For more details, see the main [Time-MoE README](../README.md#classification-fine-tuning).
