#!/usr/bin/env python3
"""
Classification data generation utilities for Time-MoE.

This module provides functions to create synthetic classification datasets
for testing and development purposes, supporting both sequence and token
classification tasks.
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import numpy as np
    from sklearn.model_selection import train_test_split

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


def _check_dependencies() -> None:
    """Check if required dependencies are available."""
    if not HAS_DEPENDENCIES:
        raise ImportError(
            "Required dependencies not found. Please install numpy and scikit-learn:\n"
            "pip install numpy scikit-learn"
        )


def create_sequence_classification_data(
    output_path: str = "sample_seq_classification.jsonl",
    num_samples: int = 100,
    seq_length: int = 512,
    num_classes: int = 3,
    seed: Optional[int] = None,
) -> None:
    """
    Create sample sequence classification data in JSONL format.

    Each sample has one label for the entire sequence.

    Parameters
    ----------
    output_path : str
        Path to save the generated data
    num_samples : int
        Number of samples to generate
    seq_length : int
        Length of each time series sequence
    num_classes : int
        Number of classification classes
    seed : int, optional
        Random seed for reproducibility
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

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
            # Class 2+: Random walk or other patterns
            if i % num_classes == 2:
                sequence = np.cumsum(0.1 * np.random.randn(seq_length))
            else:
                # Additional classes: polynomial trends
                sequence = 0.001 * t**2 + 0.1 * np.random.randn(seq_length)
            label = i % num_classes

        data.append({"sequence": sequence.tolist(), "label": label})

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Created sequence classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Number of classes: {num_classes}")


def create_token_classification_data(
    output_path: str = "sample_token_classification.jsonl",
    num_samples: int = 50,
    seq_length: int = 256,
    num_classes: int = 2,
    seed: Optional[int] = None,
) -> None:
    """
    Create sample token classification data in JSONL format.

    Each timestep in the sequence gets its own label.

    Parameters
    ----------
    output_path : str
        Path to save the generated data
    num_samples : int
        Number of samples to generate
    seq_length : int
        Length of each time series sequence
    num_classes : int
        Number of classification classes
    seed : int, optional
        Random seed for reproducibility
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

    data = []

    for i in range(num_samples):
        # Generate synthetic time series
        t = np.linspace(0, 2 * np.pi, seq_length)
        sequence = np.sin(t) + 0.1 * np.random.randn(seq_length)

        if num_classes == 2:
            # Binary classification: 0 for negative values, 1 for positive values
            labels = (sequence > 0).astype(int).tolist()
        else:
            # Multi-class: based on value ranges
            labels = []
            for val in sequence:
                if val < -0.5:
                    labels.append(0)
                elif val < 0.5:
                    labels.append(1)
                else:
                    labels.append(min(2, num_classes - 1))

        data.append({"sequence": sequence.tolist(), "label": labels})

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Created token classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Number of classes: {num_classes}")


def create_time_series_classification_data(
    num_samples: int = 1000,
    seq_length: int = 128,
    n_features: int = 1,
    n_classes: int = 3,
    seed: Optional[int] = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create sample time series classification data in the expected JSON format.

    This function creates more sophisticated time series patterns compared to
    the simpler JSONL format functions above.

    Parameters
    ----------
    num_samples : int
        Total number of samples to generate
    seq_length : int
        Length of each time series sequence
    n_features : int
        Number of features per timestep
    n_classes : int
        Number of classification classes
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple: (train_data, val_data) as lists of dictionaries
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

    data = []

    for i in range(num_samples):
        # Create different patterns for different classes
        class_label = i % n_classes

        if class_label == 0:
            # Sine wave pattern
            t = np.linspace(0, 4 * np.pi, seq_length)
            timeseries = np.sin(t) + 0.1 * np.random.randn(seq_length)
        elif class_label == 1:
            # Linear trend pattern
            timeseries = np.linspace(0, 2, seq_length) + 0.1 * np.random.randn(
                seq_length
            )
        elif class_label == 2:
            # Random walk pattern
            timeseries = np.cumsum(0.1 * np.random.randn(seq_length))
        # Additional patterns for more classes
        elif class_label == 3:
            # Exponential decay
            t = np.linspace(0, 3, seq_length)
            timeseries = np.exp(-t) + 0.1 * np.random.randn(seq_length)
        else:
            # Polynomial pattern
            t = np.linspace(-1, 1, seq_length)
            timeseries = t**3 + 0.1 * np.random.randn(seq_length)

        # Normalize
        timeseries = (timeseries - timeseries.mean()) / (timeseries.std() + 1e-8)

        # Convert to required format: [seq_len, n_features]
        if n_features == 1:
            timeseries = timeseries.reshape(-1, 1)
        else:
            # For multivariate, replicate and add noise
            timeseries = np.column_stack(
                [
                    timeseries + 0.05 * np.random.randn(seq_length)
                    for _ in range(n_features)
                ]
            )

        sample = {"timeseries": timeseries.tolist(), "label": class_label}
        data.append(sample)

    # Split into train and validation
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=[d["label"] for d in data]
    )

    return train_data, val_data


def create_token_classification_regime_data(
    num_samples: int = 500,
    seq_length: int = 128,
    n_features: int = 1,
    seed: Optional[int] = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create sample time series token classification data for regime detection.

    Each timestep gets a label indicating the current regime.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    seq_length : int
        Length of each time series sequence
    n_features : int
        Number of features per timestep
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple: (train_data, val_data) as lists of dictionaries
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

    data = []

    for i in range(num_samples):
        # Create time series with different regimes
        timeseries = []
        labels = []

        # Divide sequence into segments with different patterns
        segment_length = seq_length // 4

        for segment in range(4):
            start_idx = segment * segment_length
            end_idx = min(start_idx + segment_length, seq_length)
            length = end_idx - start_idx

            if segment == 0:
                # Normal regime (label 1)
                segment_data = 0.1 * np.random.randn(length)
                segment_labels = [1] * length
            elif segment == 1:
                # Trending regime (label 2)
                segment_data = np.linspace(0, 1, length) + 0.05 * np.random.randn(
                    length
                )
                segment_labels = [2] * length
            elif segment == 2:
                # Oscillating regime (label 3)
                t = np.linspace(0, 2 * np.pi, length)
                segment_data = np.sin(t) + 0.05 * np.random.randn(length)
                segment_labels = [3] * length
            else:
                # Anomalous regime (label 4)
                segment_data = 2 + 0.3 * np.random.randn(length)
                segment_labels = [4] * length

            timeseries.extend(segment_data.tolist())
            labels.extend(segment_labels)

        # Ensure exact length
        timeseries = timeseries[:seq_length]
        labels = labels[:seq_length]

        # Normalize timeseries
        timeseries = np.array(timeseries)
        timeseries = (timeseries - timeseries.mean()) / (timeseries.std() + 1e-8)

        # Convert to required format
        if n_features == 1:
            timeseries = timeseries.reshape(-1, 1)
        else:
            # For multivariate, replicate and add noise
            timeseries = np.column_stack(
                [
                    timeseries + 0.05 * np.random.randn(seq_length)
                    for _ in range(n_features)
                ]
            )

        sample = {"timeseries": timeseries.tolist(), "label": labels}
        data.append(sample)

    # Split into train and validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=seed)

    return train_data, val_data


def save_classification_data(data: list[dict], filename: str) -> None:
    """
    Save classification data to JSON file.

    Parameters
    ----------
    data : list
        List of data samples
    filename : str
        Output filename
    """
    os.makedirs(
        os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True
    )

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filename}")


def generate_sample_datasets(
    output_dir: str = "sample_data", seed: Optional[int] = 42
) -> None:
    """
    Generate a complete set of sample classification datasets.

    This function creates both JSONL and JSON format datasets for testing
    different classification scenarios.

    Parameters
    ----------
    output_dir : str
        Directory to save all generated datasets
    seed : int, optional
        Random seed for reproducibility
    """
    _check_dependencies()

    print("Generating sample classification datasets...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate JSONL format datasets (simpler format)
    print("\n1. Creating JSONL format datasets...")

    create_sequence_classification_data(
        output_path=os.path.join(output_dir, "sequence_classification.jsonl"),
        num_samples=200,
        seq_length=512,
        num_classes=3,
        seed=seed,
    )

    create_token_classification_data(
        output_path=os.path.join(output_dir, "token_classification.jsonl"),
        num_samples=100,
        seq_length=256,
        num_classes=2,
        seed=seed,
    )

    # Generate JSON format datasets (more sophisticated)
    print("\n2. Creating JSON format datasets...")

    # Sequence classification
    train_seq, val_seq = create_time_series_classification_data(
        num_samples=1000, seq_length=128, n_features=1, n_classes=3, seed=seed
    )
    save_classification_data(
        train_seq, os.path.join(output_dir, "train_sequence_classification.json")
    )
    save_classification_data(
        val_seq, os.path.join(output_dir, "val_sequence_classification.json")
    )

    print(
        f"Sequence classification: {len(train_seq)} train, {len(val_seq)} val samples"
    )
    print(f"Classes: {set(d['label'] for d in train_seq)}")
    print(f"Sample timeseries shape: {np.array(train_seq[0]['timeseries']).shape}")

    # Token classification
    train_tok, val_tok = create_token_classification_regime_data(
        num_samples=500, seq_length=128, n_features=1, seed=seed
    )
    save_classification_data(
        train_tok, os.path.join(output_dir, "train_token_classification.json")
    )
    save_classification_data(
        val_tok, os.path.join(output_dir, "val_token_classification.json")
    )

    print(f"Token classification: {len(train_tok)} train, {len(val_tok)} val samples")
    print(f"Classes: {set().union(*[d['label'] for d in train_tok])}")
    print(f"Sample timeseries shape: {np.array(train_tok[0]['timeseries']).shape}")
    print(f"Sample labels length: {len(train_tok[0]['label'])}")

    print(f"\nAll datasets saved to: {output_dir}")
    print("\nSample usage commands:")
    print("\n# JSONL format (simpler):")
    print(
        f"python main.py -d {output_dir}/sequence_classification.jsonl --task_type sequence_classification --num_classes 3 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
    print(
        f"python main.py -d {output_dir}/token_classification.jsonl --task_type token_classification --num_classes 2 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )

    print("\n# JSON format (with train/val split):")
    print("# Sequence Classification:")
    print("python finetune_classification.py \\")
    print("  --model_path Maple728/TimeMoE-50M \\")
    print("  --task_type sequence \\")
    print("  --num_classes 3 \\")
    print(f"  --train_data {output_dir}/train_sequence_classification.json \\")
    print(f"  --eval_data {output_dir}/val_sequence_classification.json \\")
    print("  --output_dir ./sequence_classification_output \\")
    print("  --num_epochs 5 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 2e-5")


if __name__ == "__main__":
    # Generate sample datasets when run directly
    print("Time-MoE Classification Data Generator")
    print("=" * 40)
    print()

    try:
        generate_sample_datasets()
        print()
        print("✅ Sample datasets generated successfully!")
        print()
        print("Usage examples:")
        print("# Sequence classification:")
        print("python main.py -d sample_data/sequence_classification.jsonl \\")
        print("  --task_type sequence_classification --num_classes 3")
        print()
        print("# Token classification:")
        print("python main.py -d sample_data/token_classification.jsonl \\")
        print("  --task_type token_classification --num_classes 2")

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure you have the required dependencies installed:")
        print("pip install numpy scikit-learn")
