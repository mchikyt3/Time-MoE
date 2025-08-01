#!/usr/bin/env python3
"""
Sample script to create classification datasets for testing Time-MoE classification functionality.
"""

import json
import os

import numpy as np


def create_sequence_classification_data(
    output_path="sample_seq_classification.jsonl",
    num_samples=100,
    seq_length=512,
    num_classes=3,
):
    """Create sample sequence classification data."""
    data = []

    for i in range(num_samples):
        # Generate synthetic time series with different patterns for different classes
        t = np.linspace(0, 4 * np.pi, seq_length)

        if i % num_classes == 0:
            # Class 0: Sine wave with noise
            sequence = np.sin(t) + 0.1 * np.random.randn(seq_length)
            label = 0
        elif i % num_classes == 1:
            # Class 1: Cosine wave with trend
            sequence = np.cos(t) + 0.02 * t + 0.1 * np.random.randn(seq_length)
            label = 1
        else:
            # Class 2: Random walk
            sequence = np.cumsum(0.1 * np.random.randn(seq_length))
            label = 2

        data.append({"sequence": sequence.tolist(), "label": label})

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Created sequence classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Number of classes: {num_classes}")


def create_token_classification_data(
    output_path="sample_token_classification.jsonl",
    num_samples=50,
    seq_length=256,
    num_classes=2,
):
    """Create sample token classification data."""
    data = []

    for i in range(num_samples):
        # Generate synthetic time series
        t = np.linspace(0, 2 * np.pi, seq_length)
        sequence = np.sin(t) + 0.1 * np.random.randn(seq_length)

        # Create labels: 0 for negative values, 1 for positive values
        labels = (sequence > 0).astype(int).tolist()

        data.append({"sequence": sequence.tolist(), "label": labels})

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Created token classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Number of classes: {num_classes}")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("sample_data", exist_ok=True)

    # Create sequence classification data
    create_sequence_classification_data(
        output_path="sample_data/sequence_classification.jsonl",
        num_samples=200,
        seq_length=512,
        num_classes=3,
    )

    # Create token classification data
    create_token_classification_data(
        output_path="sample_data/token_classification.jsonl",
        num_samples=100,
        seq_length=256,
        num_classes=2,
    )

    print("\nSample datasets created successfully!")
    print("You can now test classification fine-tuning with:")
    print(
        "python main.py -d sample_data/sequence_classification.jsonl --task_type sequence_classification --num_classes 3 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
    print(
        "python main.py -d sample_data/token_classification.jsonl --task_type token_classification --num_classes 2 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
