#!/usr/bin/env python3
"""
Unit tests to verify freeze_backbone functionality for all task types.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from time_moe.models.configuration_time_moe import TimeMoeConfig
    from time_moe.models.modeling_time_moe import (
        TimeMoeForPrediction,
        TimeMoeForSequenceClassification,
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestFreezeBackbone(unittest.TestCase):
    """Test freeze_backbone functionality for all task types."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_freeze_backbone_forecasting(self):
        """Test that freeze_backbone works for forecasting models."""

        # Create a small config for testing
        config = TimeMoeConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            horizon_lengths=[1, 2],
            input_size=1,
            max_position_embeddings=128,
            num_experts=2,
            num_experts_per_tok=1,
        )

        # Create forecasting model
        model = TimeMoeForPrediction(config)

        # Count initial parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(total_params, trainable_params,
                        "All parameters should be trainable initially")

        # Freeze backbone (model.model contains the TimeMoeModel backbone)
        for param in model.model.parameters():
            param.requires_grad = False

        # Count parameters after freezing
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in model.model.parameters())
        output_layer_params = sum(p.numel() for p in model.lm_heads.parameters())

        # Verify that only output layers are trainable
        self.assertEqual(trainable_after, output_layer_params,
                        f"Only output layers should be trainable, got {trainable_after} vs {output_layer_params}")
        self.assertLess(trainable_after, total_params, "Some parameters should be frozen")

    def test_freeze_backbone_classification(self):
        """Test that freeze_backbone works for classification models."""

        # Create a small config for testing
        config = TimeMoeConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            input_size=1,
            max_position_embeddings=128,
            num_experts=2,
            num_experts_per_tok=1,
            num_classes=3,
        )

        # Create classification model
        model = TimeMoeForSequenceClassification(config)

        # Count initial parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(total_params, trainable_params,
                        "All parameters should be trainable initially")

        # Freeze backbone (model.model contains the TimeMoeModel backbone)
        for param in model.model.parameters():
            param.requires_grad = False

        # Count parameters after freezing
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        classification_head_params = sum(
            p.numel() for p in model.classification_head.parameters()
        )

        # Verify that only classification head is trainable
        self.assertEqual(trainable_after, classification_head_params,
                        f"Only classification head should be trainable, got {trainable_after} vs {classification_head_params}")
        self.assertLess(trainable_after, total_params, "Some parameters should be frozen")

    def test_parameter_ratio(self):
        """Test that backbone contains most parameters, validating our freeze approach."""

        config = TimeMoeConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            horizon_lengths=[1, 4],
            input_size=1,
            max_position_embeddings=256,
            num_experts=4,
            num_experts_per_tok=2,
            num_classes=5,
        )

        # Test forecasting model
        forecasting_model = TimeMoeForPrediction(config)
        total_forecasting = sum(p.numel() for p in forecasting_model.parameters())
        backbone_forecasting = sum(p.numel() for p in forecasting_model.model.parameters())

        backbone_ratio_forecasting = backbone_forecasting / total_forecasting * 100

        # Test classification model
        classification_model = TimeMoeForSequenceClassification(config)
        total_classification = sum(p.numel() for p in classification_model.parameters())
        backbone_classification = sum(
            p.numel() for p in classification_model.model.parameters()
        )

        backbone_ratio_classification = backbone_classification / total_classification * 100

        # Backbone should be the majority of parameters
        min_backbone_percentage = 80
        self.assertGreater(backbone_ratio_forecasting, min_backbone_percentage,
                          f"Backbone should be >{min_backbone_percentage}% of parameters in forecasting, "
                          f"got {backbone_ratio_forecasting:.1f}%")
        self.assertGreater(backbone_ratio_classification, min_backbone_percentage,
                          f"Backbone should be >{min_backbone_percentage}% of parameters in classification, "
                          f"got {backbone_ratio_classification:.1f}%")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
