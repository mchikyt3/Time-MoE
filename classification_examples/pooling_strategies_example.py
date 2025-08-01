#!/usr/bin/env python3
"""
Example demonstrating different pooling strategies for Time-MoE sequence classification.

This script shows how different pooling strategies can affect classification performance
on various types of time series patterns.
"""

import json
import os

import numpy as np


def create_pooling_test_datasets():
    """Create test datasets that highlight different pooling strategy strengths."""
    os.makedirs("classification_examples", exist_ok=True)

    print("Creating datasets to test different pooling strategies...")

    # Dataset 1: Early Pattern Detection (attention/weighted should work best)
    print("1. Creating early pattern detection dataset...")
    early_pattern_data = []
    for i in range(200):
        t = np.linspace(0, 10, 256)
        base_signal = 0.1 * np.sin(t) + 0.05 * np.random.randn(256)

        if i % 3 == 0:
            # Important pattern at the beginning
            base_signal[:50] += 2.0 * np.sin(np.linspace(0, 4 * np.pi, 50))
            label = 0  # Early burst
        elif i % 3 == 1:
            # Important pattern in the middle
            base_signal[100:150] += 1.5 * np.cos(np.linspace(0, 6 * np.pi, 50))
            label = 1  # Middle spike
        else:
            # Important pattern at the end
            base_signal[200:] += 1.0 * np.sin(np.linspace(0, 8 * np.pi, 56))
            label = 2  # Late pattern

        early_pattern_data.append({"sequence": base_signal.tolist(), "label": label})

    with open("classification_examples/early_pattern_detection.jsonl", "w") as f:
        for item in early_pattern_data:
            f.write(json.dumps(item) + "\n")

    # Dataset 2: Global Statistics (mean/max pooling should work best)
    print("2. Creating global statistics dataset...")
    global_stats_data = []
    for i in range(200):
        if i % 3 == 0:
            # High variance signal
            sequence = np.random.normal(0, 2, 256)
            label = 0  # High variance
        elif i % 3 == 1:
            # Low variance signal
            sequence = np.random.normal(0, 0.2, 256)
            label = 1  # Low variance
        else:
            # Medium variance with outliers
            sequence = np.random.normal(0, 0.5, 256)
            # Add some outliers
            outlier_indices = np.random.choice(256, size=5, replace=False)
            sequence[outlier_indices] += np.random.choice([-3, 3], size=5)
            label = 2  # With outliers

        global_stats_data.append({"sequence": sequence.tolist(), "label": label})

    with open("classification_examples/global_statistics.jsonl", "w") as f:
        for item in global_stats_data:
            f.write(json.dumps(item) + "\n")

    # Dataset 3: Temporal Evolution (last_token/weighted_temporal should work best)
    print("3. Creating temporal evolution dataset...")
    temporal_data = []
    for i in range(200):
        t = np.linspace(0, 10, 256)

        if i % 3 == 0:
            # Accelerating trend
            sequence = 0.01 * t**2 + 0.1 * np.random.randn(256)
            label = 0  # Accelerating
        elif i % 3 == 1:
            # Decelerating trend
            sequence = 2 * np.sqrt(t) + 0.1 * np.random.randn(256)
            label = 1  # Decelerating
        else:
            # Oscillating with increasing amplitude
            sequence = (0.1 * t) * np.sin(t) + 0.1 * np.random.randn(256)
            label = 2  # Oscillating

        temporal_data.append({"sequence": sequence.tolist(), "label": label})

    with open("classification_examples/temporal_evolution.jsonl", "w") as f:
        for item in temporal_data:
            f.write(json.dumps(item) + "\n")

    print("✅ Test datasets created!")
    return [
        (
            "early_pattern_detection.jsonl",
            "Early Pattern Detection",
            ["attention", "multi_scale"],
        ),
        (
            "global_statistics.jsonl",
            "Global Statistics",
            ["mean", "max", "multi_scale"],
        ),
        (
            "temporal_evolution.jsonl",
            "Temporal Evolution",
            ["last_token", "weighted_temporal", "conv_pool"],
        ),
    ]


def print_pooling_strategy_guide():
    """Print guide on when to use different pooling strategies."""
    print("\n🎯 Pooling Strategy Guide:")
    print("=" * 60)

    strategies = [
        {
            "name": "last_token",
            "description": "Uses the final timestep representation",
            "best_for": [
                "Sequential patterns",
                "Temporal dependencies",
                "State evolution",
            ],
            "example": "Stock price trends, speech recognition final states",
        },
        {
            "name": "mean",
            "description": "Average of all timestep representations",
            "best_for": [
                "Global statistics",
                "Overall signal characteristics",
                "Stable patterns",
            ],
            "example": "Average heart rate, overall sentiment analysis",
        },
        {
            "name": "max",
            "description": "Maximum activation across all timesteps",
            "best_for": ["Peak detection", "Anomaly identification", "Spike patterns"],
            "example": "Seizure detection, network intrusion detection",
        },
        {
            "name": "attention",
            "description": "Learns which timesteps are most important",
            "best_for": [
                "Variable-length patterns",
                "Complex temporal relationships",
                "Multi-scale features",
            ],
            "example": "Document classification, irregular heartbeats",
        },
        {
            "name": "multi_scale",
            "description": "Combines mean, max, and last token",
            "best_for": [
                "Complex patterns",
                "Robust classification",
                "Unknown pattern types",
            ],
            "example": "General-purpose classification, exploratory analysis",
        },
        {
            "name": "weighted_temporal",
            "description": "Recent timesteps weighted more heavily",
            "best_for": [
                "Recent events more important",
                "Trend analysis",
                "Recency bias",
            ],
            "example": "Real-time monitoring, recent trend classification",
        },
        {
            "name": "conv_pool",
            "description": "1D convolution followed by pooling",
            "best_for": [
                "Local patterns",
                "Feature extraction",
                "Translation invariance",
            ],
            "example": "Pattern recognition, motif detection",
        },
    ]

    for strategy in strategies:
        print(f"\n📊 {strategy['name'].upper()}")
        print(f"   Description: {strategy['description']}")
        print(f"   Best for: {', '.join(strategy['best_for'])}")
        print(f"   Example: {strategy['example']}")


def print_training_examples():
    """Print training command examples for different pooling strategies."""
    datasets = create_pooling_test_datasets()

    print("\n🚀 Training Examples:")
    print("=" * 60)

    for dataset_file, dataset_name, recommended_strategies in datasets:
        print(f"\n📁 {dataset_name}")
        print(f"Dataset: classification_examples/{dataset_file}")
        print(f"Recommended strategies: {', '.join(recommended_strategies)}")

        for strategy in recommended_strategies:
            print(f"\n  🔹 Using {strategy} pooling:")
            print("    python main.py \\")
            print(f"      -d classification_examples/{dataset_file} \\")
            print("      --task_type sequence_classification \\")
            print("      --num_classes 3 \\")
            print(f"      --pooling_strategy {strategy} \\")
            print("      --micro_batch_size 4 \\")
            print("      --train_steps 100 \\")
            print("      --learning_rate 1e-4 \\")
            print("      --precision bf16")


def print_performance_tips():
    """Print tips for optimizing pooling strategy performance."""
    print("\n💡 Performance Tips:")
    print("=" * 60)

    tips = [
        "🔍 Try multiple strategies: Different datasets may benefit from different approaches",
        "📊 Use multi_scale as baseline: Combines multiple pooling methods for robust performance",
        "⚡ Start with simple strategies: last_token and mean are fast and often effective",
        "🎯 Consider your domain: Medical signals vs. financial data may need different strategies",
        "📈 Monitor validation metrics: Use early stopping to compare strategies fairly",
        "🔄 Ensemble different strategies: Train multiple models and combine predictions",
        "⚙️ Tune hyperparameters: Learning rate and dropout may need adjustment per strategy",
        "📝 Log attention weights: For attention pooling, visualize which timesteps matter most",
    ]

    for tip in tips:
        print(f"  {tip}")

    print("\n🔬 Experimental Setup:")
    print("  • Use the same train/validation split for fair comparison")
    print("  • Run multiple seeds and report average performance")
    print("  • Consider computational cost vs. accuracy trade-offs")
    print("  • Validate on held-out test set for final evaluation")


if __name__ == "__main__":
    print("🔥 Time-MoE Pooling Strategies Guide")
    print("=" * 60)

    print_pooling_strategy_guide()
    print_training_examples()
    print_performance_tips()

    print("\n✨ Summary:")
    print("Different pooling strategies capture different aspects of time series:")
    print("• Global patterns → mean, max pooling")
    print("• Temporal evolution → last_token, weighted_temporal")
    print("• Complex patterns → attention, multi_scale")
    print("• Local features → conv_pool")
    print(
        "\nExperiment with multiple strategies to find the best for your specific task!"
    )
