#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Inference script for Time-MoE classification models
"""

import argparse
import json
from typing import List, Union

import numpy as np
import torch
from time_moe.models.modeling_time_moe_classification import (
    TimeMoeForSequenceClassification,
    TimeMoeForTokenClassification,
)


def load_classification_model(model_path: str, task_type: str):
    """Load a fine-tuned classification model"""
    if task_type == "sequence":
        model = TimeMoeForSequenceClassification.from_pretrained(model_path)
    else:
        model = TimeMoeForTokenClassification.from_pretrained(model_path)

    model.eval()
    return model


def preprocess_timeseries(timeseries: Union[List, np.ndarray], max_length: int = 1024):
    """Preprocess time series data for inference"""
    if isinstance(timeseries, list):
        timeseries = np.array(timeseries, dtype=np.float32)

    # Ensure 2D: [seq_len, n_features]
    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape(-1, 1)

    # Truncate or pad to max_length
    seq_len, n_features = timeseries.shape
    if seq_len > max_length:
        timeseries = timeseries[:max_length]
    elif seq_len < max_length:
        padding = np.zeros((max_length - seq_len, n_features), dtype=np.float32)
        timeseries = np.vstack([timeseries, padding])

    return torch.tensor(timeseries, dtype=torch.float32).unsqueeze(
        0
    )  # Add batch dimension


def predict_sequence_classification(
    model, timeseries: Union[List, np.ndarray], class_names: List[str] = None
):
    """
    Predict class for a single time series sequence

    Args:
        model: Fine-tuned TimeMoeForSequenceClassification model
        timeseries: Time series data
        class_names: Optional list of class names

    Returns:
        dict: Prediction results
    """
    # Preprocess input
    input_tensor = preprocess_timeseries(timeseries)

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Format results
    results = {
        "predicted_class": predicted_class,
        "confidence": probabilities[0, predicted_class].item(),
        "all_probabilities": probabilities[0].tolist(),
    }

    if class_names:
        results["predicted_class_name"] = class_names[predicted_class]
        results["class_probabilities"] = {
            name: prob for name, prob in zip(class_names, results["all_probabilities"])
        }

    return results


def predict_token_classification(
    model, timeseries: Union[List, np.ndarray], class_names: List[str] = None
):
    """
    Predict class for each timestep in a time series

    Args:
        model: Fine-tuned TimeMoeForTokenClassification model
        timeseries: Time series data
        class_names: Optional list of class names

    Returns:
        dict: Prediction results
    """
    # Preprocess input
    input_tensor = preprocess_timeseries(timeseries)
    original_length = (
        len(timeseries) if isinstance(timeseries, list) else timeseries.shape[0]
    )

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits  # [1, seq_len, num_classes]
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = (
            torch.argmax(logits, dim=-1).squeeze(0).tolist()
        )  # [seq_len]

    # Only return predictions for the original sequence length (exclude padding)
    predicted_classes = predicted_classes[:original_length]
    token_probabilities = probabilities[0, :original_length].tolist()

    # Format results
    results = {
        "predicted_classes": predicted_classes,
        "sequence_length": original_length,
        "token_probabilities": token_probabilities,
    }

    if class_names:
        results["predicted_class_names"] = [
            class_names[cls] for cls in predicted_classes
        ]
        results["token_predictions"] = [
            {
                "timestep": i,
                "predicted_class": cls,
                "predicted_class_name": class_names[cls] if class_names else None,
                "confidence": max(probs),
                "all_probabilities": dict(zip(class_names, probs))
                if class_names
                else probs,
            }
            for i, (cls, probs) in enumerate(
                zip(predicted_classes, token_probabilities)
            )
        ]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Time-MoE classification model"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["sequence", "token"],
        required=True,
        help="Type of classification task",
    )
    parser.add_argument(
        "--input_data", type=str, required=True, help="Path to input data (JSON file)"
    )
    parser.add_argument(
        "--class_names", type=str, nargs="+", help="Names of the classes (optional)"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save predictions (optional)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.task_type} classification model from {args.model_path}...")
    model = load_classification_model(args.model_path, args.task_type)
    print(f"Model loaded successfully. Number of classes: {model.num_classes}")

    # Load input data
    print(f"Loading input data from {args.input_data}...")
    with open(args.input_data, "r") as f:
        input_data = json.load(f)

    # Handle different input formats
    if isinstance(input_data, dict) and "timeseries" in input_data:
        # Single sample
        timeseries_data = [input_data["timeseries"]]
    elif isinstance(input_data, list):
        if all("timeseries" in item for item in input_data):
            # List of samples with metadata
            timeseries_data = [item["timeseries"] for item in input_data]
        else:
            # List of raw time series
            timeseries_data = input_data
    else:
        # Single raw time series
        timeseries_data = [input_data]

    print(f"Processing {len(timeseries_data)} time series...")

    # Run inference
    all_predictions = []

    for i, timeseries in enumerate(timeseries_data):
        print(f"Processing sample {i + 1}/{len(timeseries_data)}...")

        if args.task_type == "sequence":
            prediction = predict_sequence_classification(
                model, timeseries, args.class_names
            )
        else:
            prediction = predict_token_classification(
                model, timeseries, args.class_names
            )

        prediction["sample_id"] = i
        all_predictions.append(prediction)

        # Print results for this sample
        if args.task_type == "sequence":
            class_name = prediction.get(
                "predicted_class_name", prediction["predicted_class"]
            )
            confidence = prediction["confidence"]
            print(f"  Sample {i}: {class_name} (confidence: {confidence:.3f})")
        else:
            unique_classes = set(prediction["predicted_classes"])
            print(f"  Sample {i}: {len(unique_classes)} unique classes detected")

    # Save results if requested
    if args.output_file:
        print(f"Saving predictions to {args.output_file}...")
        with open(args.output_file, "w") as f:
            json.dump(all_predictions, f, indent=2)

    # Print summary
    print(f"\nInference completed for {len(timeseries_data)} samples.")

    if args.task_type == "sequence":
        # Summary for sequence classification
        predicted_classes = [p["predicted_class"] for p in all_predictions]
        unique_predicted = set(predicted_classes)
        print(f"Predicted classes: {sorted(unique_predicted)}")

        if args.class_names:
            for class_id in sorted(unique_predicted):
                count = predicted_classes.count(class_id)
                class_name = (
                    args.class_names[class_id]
                    if class_id < len(args.class_names)
                    else f"Class_{class_id}"
                )
                print(f"  {class_name}: {count} samples")

    else:
        # Summary for token classification
        all_token_classes = []
        for p in all_predictions:
            all_token_classes.extend(p["predicted_classes"])

        unique_token_classes = set(all_token_classes)
        print(f"Token classes found: {sorted(unique_token_classes)}")

        if args.class_names:
            for class_id in sorted(unique_token_classes):
                count = all_token_classes.count(class_id)
                class_name = (
                    args.class_names[class_id]
                    if class_id < len(args.class_names)
                    else f"Class_{class_id}"
                )
                print(f"  {class_name}: {count} tokens")


if __name__ == "__main__":
    main()
