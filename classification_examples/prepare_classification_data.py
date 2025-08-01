#!/usr/bin/env python
"""
Example script showing how to prepare data and fine-tune Time-MoE for classification
"""

import json

import numpy as np
from sklearn.model_selection import train_test_split


def create_sample_time_series_classification_data():
    """
    Create sample time series classification data in the expected format.

    Returns
    -------
        tuple: (train_data, val_data) as lists of dictionaries
    """
    np.random.seed(42)

    # Generate synthetic time series data
    n_samples = 1000
    seq_length = 128
    n_features = 1
    n_classes = 3

    data = []

    for i in range(n_samples):
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
        else:
            # Random walk pattern
            timeseries = np.cumsum(0.1 * np.random.randn(seq_length))

        # Normalize
        timeseries = (timeseries - timeseries.mean()) / (timeseries.std() + 1e-8)

        # Convert to required format: [seq_len, n_features]
        if n_features == 1:
            timeseries = timeseries.reshape(-1, 1)

        sample = {"timeseries": timeseries.tolist(), "label": class_label}
        data.append(sample)

    # Split into train and validation
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=[d["label"] for d in data]
    )

    return train_data, val_data


def create_sample_token_classification_data():
    """
    Create sample time series token classification data.
    Each timestep gets a label (e.g., for anomaly detection, regime classification, etc.)

    Returns
    -------
        tuple: (train_data, val_data) as lists of dictionaries
    """
    np.random.seed(42)

    n_samples = 500
    seq_length = 128
    n_features = 1

    data = []

    for i in range(n_samples):
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

        sample = {"timeseries": timeseries.tolist(), "label": labels}
        data.append(sample)

    # Split into train and validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    return train_data, val_data


def save_data(data, filename):
    """Save data to JSON file"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filename}")


def main():
    """Create sample datasets for both sequence and token classification"""

    print("Creating sample data for Time-MoE classification fine-tuning...")

    # Create sequence classification data
    print("\n1. Creating sequence classification data...")
    train_seq, val_seq = create_sample_time_series_classification_data()
    save_data(train_seq, "train_sequence_classification.json")
    save_data(val_seq, "val_sequence_classification.json")

    print(
        f"Sequence classification: {len(train_seq)} train, {len(val_seq)} val samples"
    )
    print(f"Classes: {set(d['label'] for d in train_seq)}")
    print(f"Sample timeseries shape: {np.array(train_seq[0]['timeseries']).shape}")

    # Create token classification data
    print("\n2. Creating token classification data...")
    train_tok, val_tok = create_sample_token_classification_data()
    save_data(train_tok, "train_token_classification.json")
    save_data(val_tok, "val_token_classification.json")

    print(f"Token classification: {len(train_tok)} train, {len(val_tok)} val samples")
    print(f"Classes: {set().union(*[d['label'] for d in train_tok])}")
    print(f"Sample timeseries shape: {np.array(train_tok[0]['timeseries']).shape}")
    print(f"Sample labels length: {len(train_tok[0]['label'])}")

    print("\nSample usage commands:")
    print("\n# Sequence Classification:")
    print("python finetune_classification.py \\")
    print("  --model_path Maple728/TimeMoE-50M \\")
    print("  --task_type sequence \\")
    print("  --num_classes 3 \\")
    print("  --train_data train_sequence_classification.json \\")
    print("  --eval_data val_sequence_classification.json \\")
    print("  --output_dir ./sequence_classification_output \\")
    print("  --num_epochs 5 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 2e-5")

    print("\n# Token Classification:")
    print("python finetune_classification.py \\")
    print("  --model_path Maple728/TimeMoE-50M \\")
    print("  --task_type token \\")
    print("  --num_classes 5 \\")  # includes padding class 0
    print("  --train_data train_token_classification.json \\")
    print("  --eval_data val_token_classification.json \\")
    print("  --output_dir ./token_classification_output \\")
    print("  --num_epochs 5 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 2e-5")

    print("\n# To freeze the backbone and only train the classification head:")
    print("# Add --freeze_backbone to either command above")

    print("\nData format:")
    print("Each JSON file contains a list of samples with:")
    print("- 'timeseries': 2D array [seq_len, n_features]")
    print("- 'label': int (sequence) or list of ints (token classification)")


if __name__ == "__main__":
    main()
