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

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


def _check_dependencies() -> None:
    """Check if required dependencies are available."""
    if not HAS_DEPENDENCIES:
        raise ImportError(
            "Required dependencies not found. Please install numpy:\npip install numpy"
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


def create_sequence_classification_jsonl(
    output_path: str = "sequence_classification.jsonl",
    num_samples: int = 1000,
    seq_length: int = 128,
    n_classes: int = 3,
    seed: Optional[int] = 42,
) -> None:
    """
    Create sample time series sequence classification data in JSONL format.

    This function creates more sophisticated time series patterns with different
    regimes for each class.

    Parameters
    ----------
    output_path : str
        Path to save the generated JSONL data
    num_samples : int
        Total number of samples to generate
    seq_length : int
        Length of each time series sequence
    n_classes : int
        Number of classification classes
    seed : int, optional
        Random seed for reproducibility
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w") as f:
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

            sample = {"sequence": timeseries.tolist(), "label": class_label}
            f.write(json.dumps(sample) + "\n")

    print(f"Created sophisticated sequence classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print(f"- Number of classes: {n_classes}")


def create_token_classification_regime_jsonl(
    output_path: str = "token_classification.jsonl",
    num_samples: int = 500,
    seq_length: int = 128,
    seed: Optional[int] = 42,
) -> None:
    """
    Create sample time series token classification data for regime detection in JSONL format.

    Each timestep gets a label indicating the current regime.

    Parameters
    ----------
    output_path : str
        Path to save the generated JSONL data
    num_samples : int
        Number of samples to generate
    seq_length : int
        Length of each time series sequence
    seed : int, optional
        Random seed for reproducibility
    """
    _check_dependencies()

    if seed is not None:
        np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w") as f:
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

            sample = {"sequence": timeseries.tolist(), "label": labels}
            f.write(json.dumps(sample) + "\n")

    print(f"Created sophisticated token classification dataset: {output_path}")
    print(f"- Number of samples: {num_samples}")
    print(f"- Sequence length: {seq_length}")
    print("- Classes: [1, 2, 3, 4] (Normal, Trending, Oscillating, Anomalous)")


def generate_sample_datasets(
    output_dir: str = "sample_data", seed: Optional[int] = 42
) -> None:
    """
    Generate a complete set of sample classification datasets in JSONL format.

    This function creates JSONL format datasets for testing different
    classification scenarios.

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

    # Generate JSONL format datasets
    print("\nCreating JSONL format datasets...")

    # Simple sequence classification
    create_sequence_classification_data(
        output_path=os.path.join(output_dir, "simple_sequence_classification.jsonl"),
        num_samples=200,
        seq_length=512,
        num_classes=3,
        seed=seed,
    )

    # Simple token classification
    create_token_classification_data(
        output_path=os.path.join(output_dir, "simple_token_classification.jsonl"),
        num_samples=100,
        seq_length=256,
        num_classes=2,
        seed=seed,
    )

    # Sophisticated sequence classification
    create_sequence_classification_jsonl(
        output_path=os.path.join(output_dir, "sequence_classification.jsonl"),
        num_samples=1000,
        seq_length=128,
        n_classes=3,
        seed=seed,
    )

    # Sophisticated token classification (regime detection)
    create_token_classification_regime_jsonl(
        output_path=os.path.join(output_dir, "token_classification.jsonl"),
        num_samples=500,
        seq_length=128,
        seed=seed,
    )

    print(f"\nAll datasets saved to: {output_dir}")
    print("\nSample usage commands:")
    print("\n# Simple sequence classification:")
    print(
        f"python main.py -d {output_dir}/simple_sequence_classification.jsonl --task_type sequence_classification --num_classes 3 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
    print("\n# Simple token classification:")
    print(
        f"python main.py -d {output_dir}/simple_token_classification.jsonl --task_type token_classification --num_classes 2 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
    print("\n# Sophisticated sequence classification:")
    print(
        f"python main.py -d {output_dir}/sequence_classification.jsonl --task_type sequence_classification --num_classes 3 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )
    print("\n# Sophisticated token classification (regime detection):")
    print(
        f"python main.py -d {output_dir}/token_classification.jsonl --task_type token_classification --num_classes 5 --micro_batch_size 2 --train_steps 10 --attn_implementation eager --precision bf16"
    )


if __name__ == "__main__":
    # Generate sample datasets when run directly
    print("Time-MoE Classification Data Generator")
    print("=" * 40)
    print()

    try:
        generate_sample_datasets()
        print("✅ Sample datasets generated successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure you have the required dependencies installed:")
        print("pip install numpy")
