#!/usr/bin/env python3
"""
Complete example of Time-MoE classification fine-tuning.

This script demonstrates how to:
1. Create classification datasets
2. Fine-tune Time-MoE for sequence classification
3. Fine-tune Time-MoE for token classification
4. Use backbone freezing for efficient training
"""

import json
import os

import numpy as np


def create_example_datasets():
    """Create example classification datasets."""
    os.makedirs("classification_examples", exist_ok=True)

    # Sequence Classification Example: Trend Detection
    print("Creating sequence classification dataset (trend detection)...")
    seq_data = []
    for i in range(150):
        t = np.linspace(0, 10, 256)
        if i % 3 == 0:
            # Upward trend
            sequence = 0.1 * t + 0.1 * np.sin(t) + 0.05 * np.random.randn(256)
            label = 0  # Upward
        elif i % 3 == 1:
            # Downward trend
            sequence = -0.1 * t + 0.1 * np.sin(t) + 0.05 * np.random.randn(256)
            label = 1  # Downward
        else:
            # No trend
            sequence = 0.1 * np.sin(t) + 0.05 * np.random.randn(256)
            label = 2  # Flat

        seq_data.append({"sequence": sequence.tolist(), "label": label})

    with open("classification_examples/trend_detection.jsonl", "w") as f:
        for item in seq_data:
            f.write(json.dumps(item) + "\n")

    # Token Classification Example: Anomaly Detection
    print("Creating token classification dataset (anomaly detection)...")
    token_data = []
    for i in range(100):
        t = np.linspace(0, 4 * np.pi, 128)
        base_signal = np.sin(t) + 0.1 * np.random.randn(128)

        # Add random anomalies
        anomaly_indices = np.random.choice(
            128, size=np.random.randint(5, 15), replace=False
        )
        base_signal[anomaly_indices] += np.random.normal(0, 2, len(anomaly_indices))

        # Label: 1 for anomalies, 0 for normal
        labels = np.zeros(128, dtype=int)
        labels[anomaly_indices] = 1

        token_data.append({"sequence": base_signal.tolist(), "label": labels.tolist()})

    with open("classification_examples/anomaly_detection.jsonl", "w") as f:
        for item in token_data:
            f.write(json.dumps(item) + "\n")

    print("✅ Example datasets created!")
    print(
        "- classification_examples/trend_detection.jsonl (3 classes: upward, downward, flat)"
    )
    print(
        "- classification_examples/anomaly_detection.jsonl (2 classes: normal, anomaly)"
    )


def print_training_commands():
    """Print example training commands."""
    print("\n🚀 Example Training Commands:")
    print("\n1. Sequence Classification (Trend Detection):")
    print("   python main.py -d classification_examples/trend_detection.jsonl \\")
    print("     --task_type sequence_classification \\")
    print("     --num_classes 3 \\")
    print("     --micro_batch_size 4 \\")
    print("     --train_steps 50 \\")
    print("     --learning_rate 1e-4 \\")
    print("     --precision bf16 \\")
    print("     --attn_implementation eager")

    print("\n2. Token Classification (Anomaly Detection):")
    print("   python main.py -d classification_examples/anomaly_detection.jsonl \\")
    print("     --task_type token_classification \\")
    print("     --num_classes 2 \\")
    print("     --micro_batch_size 4 \\")
    print("     --train_steps 50 \\")
    print("     --learning_rate 1e-4 \\")
    print("     --precision bf16 \\")
    print("     --attn_implementation eager")

    print("\n3. With Backbone Freezing (for small datasets):")
    print("   python main.py -d classification_examples/trend_detection.jsonl \\")
    print("     --task_type sequence_classification \\")
    print("     --num_classes 3 \\")
    print("     --freeze_backbone \\")
    print("     --micro_batch_size 4 \\")
    print("     --train_steps 30 \\")
    print("     --learning_rate 1e-3 \\")
    print("     --precision bf16 \\")
    print("     --attn_implementation eager")

    print("\n4. Multi-GPU Training:")
    print(
        "   python torch_dist_run.py main.py -d classification_examples/trend_detection.jsonl \\"
    )
    print("     --task_type sequence_classification \\")
    print("     --num_classes 3 \\")
    print("     --global_batch_size 32 \\")
    print("     --train_steps 100 \\")
    print("     --precision bf16 \\")
    print("     --attn_implementation eager")


def print_data_format_info():
    """Print information about data formats."""
    print("\n📋 Data Format Requirements:")
    print("\nSequence Classification:")
    print('{"sequence": [1.0, 2.0, 3.0, ...], "label": 0}')
    print('{"sequence": [4.0, 5.0, 6.0, ...], "label": 1}')
    print("- Each sequence gets ONE label")
    print("- Labels should be integers: 0, 1, 2, ...")

    print("\nToken Classification:")
    print('{"sequence": [1.0, 2.0, 3.0, ...], "label": [0, 1, 0, ...]}')
    print('{"sequence": [4.0, 5.0, 6.0, ...], "label": [1, 0, 1, ...]}')
    print("- Each timestep gets ONE label")
    print("- Labels should be lists of integers matching sequence length")
    print("- Use -100 for positions to ignore in loss calculation")


if __name__ == "__main__":
    print("🔥 Time-MoE Classification Fine-tuning Example")
    print("=" * 50)

    create_example_datasets()
    print_data_format_info()
    print_training_commands()

    print("\n✨ Key Features:")
    print("- Transfer learning from pretrained Time-MoE models")
    print("- Sequence classification (one label per series)")
    print("- Token classification (one label per timestep)")
    print("- Backbone freezing for small datasets")
    print("- Support for JSONL, JSON, and pickle formats")
    print("- Multi-GPU training with torch_dist_run.py")

    print("\n📚 For more information, see the updated README.md!")
