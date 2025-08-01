#!/usr/bin/env python3
"""
Integration tests for Time-MoE package.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from time_moe.models.configuration_time_moe import TimeMoeConfig
    from time_moe.models.modeling_time_moe import (
        TimeMoeForPrediction,
        TimeMoeForSequenceClassification,
        TimeMoeForTokenClassification,
    )
    from time_moe.datasets.general_dataset import GeneralDataset
    from time_moe.runner import TimeMoeRunner

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestTimeMoeIntegration(unittest.TestCase):
    """Integration tests for Time-MoE components working together."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        # Create minimal config for fast testing
        self.config = TimeMoeConfig(
            input_size=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            horizon_lengths=[1],
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=32,
        )

    def test_forecasting_end_to_end(self):
        """Test end-to-end forecasting workflow."""
        # Create model
        model = TimeMoeForPrediction(self.config)
        model.eval()

        # Create sample input (raw time series data)
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Check outputs
        self.assertIsNotNone(outputs.logits)
        # For single horizon prediction, output should be [batch_size, seq_len, input_size * horizon_length]
        expected_shape = (
            batch_size,
            seq_len,
            self.config.input_size * self.config.horizon_lengths[0],
        )
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_sequence_classification_end_to_end(self):
        """Test end-to-end sequence classification workflow."""
        # Create classification config
        config = TimeMoeConfig(
            input_size=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=32,
            num_classes=3,
            classifier_dropout=0.1,
        )

        # Create model
        model = TimeMoeForSequenceClassification(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, config.input_size)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Check outputs
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (batch_size, config.num_classes))

    def test_token_classification_end_to_end(self):
        """Test end-to-end token classification workflow."""
        # Create classification config
        config = TimeMoeConfig(
            input_size=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=32,
            num_classes=4,
            classifier_dropout=0.1,
        )

        # Create model
        model = TimeMoeForTokenClassification(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, config.input_size)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Check outputs
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(
            outputs.logits.shape, (batch_size, seq_len, config.num_classes)
        )

    def test_model_training_mode_switch(self):
        """Test switching between training and evaluation modes."""
        model = TimeMoeForPrediction(self.config)

        # Test training mode
        model.train()
        self.assertTrue(model.training)

        batch_size, seq_len = 1, 4
        input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

        outputs_train = model(input_ids=input_ids)
        self.assertIsNotNone(outputs_train.logits)

        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)

        with torch.no_grad():
            outputs_eval = model(input_ids=input_ids)

        self.assertIsNotNone(outputs_eval.logits)
        self.assertEqual(outputs_train.logits.shape, outputs_eval.logits.shape)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        model = TimeMoeForPrediction(self.config)
        model.train()

        # Create sample input and target
        batch_size, seq_len = 1, 4
        input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

        # Forward pass
        outputs = model(input_ids=input_ids)

        # Create target with correct shape matching output
        target = torch.randn_like(outputs.logits)

        # Compute loss
        loss = torch.nn.functional.mse_loss(outputs.logits, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        self.assertTrue(has_grad, "Model should have gradients after backward pass")

    def test_config_serialization_roundtrip(self):
        """Test that config can be saved and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config"

            # Save config
            self.config.save_pretrained(config_path)

            # Load config
            loaded_config = TimeMoeConfig.from_pretrained(config_path)

            # Compare key attributes
            self.assertEqual(loaded_config.input_size, self.config.input_size)
            self.assertEqual(loaded_config.hidden_size, self.config.hidden_size)
            self.assertEqual(loaded_config.num_experts, self.config.num_experts)
            self.assertEqual(loaded_config.horizon_lengths, self.config.horizon_lengths)

    def test_runner_model_loading(self):
        """Test runner model loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config"

            # Save config
            self.config.save_pretrained(config_path)

            # Create runner
            runner = TimeMoeRunner(model_path=str(config_path))

            # Load forecasting model
            model = runner.load_model(
                from_scratch=True, task_type="forecasting", attn_implementation="eager"
            )

            self.assertIsNotNone(model)
            self.assertEqual(type(model).__name__, "TimeMoeForPrediction")

            # Test model works
            batch_size, seq_len = 1, 4
            input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            self.assertIsNotNone(outputs.logits)

    def test_multi_horizon_forecasting(self):
        """Test forecasting with multiple horizons."""
        # Create config with multiple horizons
        config = TimeMoeConfig(
            input_size=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            horizon_lengths=[1, 2, 4],  # Multiple horizons
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=32,
        )

        model = TimeMoeForPrediction(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, config.input_size)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Check outputs shape includes all horizons
        # For inference, it uses the first horizon by default
        expected_shape = (
            batch_size,
            seq_len,
            config.input_size * config.horizon_lengths[0],  # Uses first horizon
        )
        self.assertEqual(outputs.logits.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
