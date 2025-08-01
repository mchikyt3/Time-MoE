#!/usr/bin/env python3
"""
Unit tests for TimeMoeConfig.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from time_moe.models.configuration_time_moe import TimeMoeConfig

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestTimeMoeConfig(unittest.TestCase):
    """Test TimeMoeConfig functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_default_config(self):
        """Test default configuration values."""
        config = TimeMoeConfig()

        # Test default values
        self.assertEqual(config.input_size, 1)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 22016)
        self.assertEqual(config.horizon_lengths, [1])  # Converted to list
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(
            config.num_key_value_heads, 32
        )  # Defaults to num_attention_heads
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.num_experts, 1)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertAlmostEqual(config.initializer_range, 0.02)
        self.assertAlmostEqual(config.rms_norm_eps, 1e-6)
        self.assertTrue(config.use_cache)
        self.assertFalse(config.use_dense)
        self.assertEqual(config.rope_theta, 10000)
        self.assertAlmostEqual(config.attention_dropout, 0.0)
        self.assertTrue(config.apply_aux_loss)
        self.assertAlmostEqual(config.router_aux_loss_factor, 0.02)
        self.assertFalse(config.tie_word_embeddings)
        self.assertIsNone(config.num_classes)
        self.assertAlmostEqual(config.classifier_dropout, 0.1)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimeMoeConfig(
            input_size=10,
            hidden_size=128,
            intermediate_size=256,
            horizon_lengths=[1, 2, 3],
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_act="relu",
            num_experts_per_tok=3,
            num_experts=8,
            max_position_embeddings=1024,
            initializer_range=0.05,
            rms_norm_eps=1e-5,
            use_cache=False,
            use_dense=True,
            rope_theta=20000,
            attention_dropout=0.1,
            apply_aux_loss=False,
            router_aux_loss_factor=0.05,
            num_classes=5,
            classifier_dropout=0.2,
        )

        self.assertEqual(config.input_size, 10)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.intermediate_size, 256)
        self.assertEqual(config.horizon_lengths, [1, 2, 3])
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.hidden_act, "relu")
        self.assertEqual(config.num_experts_per_tok, 3)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.max_position_embeddings, 1024)
        self.assertAlmostEqual(config.initializer_range, 0.05)
        self.assertAlmostEqual(config.rms_norm_eps, 1e-5)
        self.assertFalse(config.use_cache)
        self.assertTrue(config.use_dense)
        self.assertEqual(config.rope_theta, 20000)
        self.assertAlmostEqual(config.attention_dropout, 0.1)
        self.assertFalse(config.apply_aux_loss)
        self.assertAlmostEqual(config.router_aux_loss_factor, 0.05)
        self.assertEqual(config.num_classes, 5)
        self.assertAlmostEqual(config.classifier_dropout, 0.2)

    def test_horizon_lengths_conversion(self):
        """Test that single int horizon_lengths is converted to list."""
        config = TimeMoeConfig(horizon_lengths=5)
        self.assertEqual(config.horizon_lengths, [5])

        config = TimeMoeConfig(horizon_lengths=[1, 2, 3])
        self.assertEqual(config.horizon_lengths, [1, 2, 3])

    def test_num_key_value_heads_default(self):
        """Test that num_key_value_heads defaults to num_attention_heads."""
        config = TimeMoeConfig(num_attention_heads=16)
        self.assertEqual(config.num_key_value_heads, 16)

        config = TimeMoeConfig(num_attention_heads=16, num_key_value_heads=8)
        self.assertEqual(config.num_key_value_heads, 8)

    def test_use_dense_apply_aux_loss_assertion(self):
        """Test that use_dense and apply_aux_loss cannot both be True or both be False."""
        # Valid combinations
        TimeMoeConfig(use_dense=True, apply_aux_loss=False)  # Should not raise
        TimeMoeConfig(use_dense=False, apply_aux_loss=True)  # Should not raise

        # Invalid combinations should raise AssertionError
        with self.assertRaises(AssertionError):
            TimeMoeConfig(use_dense=True, apply_aux_loss=True)

        with self.assertRaises(AssertionError):
            TimeMoeConfig(use_dense=False, apply_aux_loss=False)

    def test_model_type(self):
        """Test that model_type is correctly set."""
        config = TimeMoeConfig()
        self.assertEqual(config.model_type, "time_moe")

    def test_keys_to_ignore_at_inference(self):
        """Test that keys_to_ignore_at_inference is correctly set."""
        config = TimeMoeConfig()
        self.assertEqual(config.keys_to_ignore_at_inference, ["past_key_values"])

    def test_classification_config(self):
        """Test configuration for classification tasks."""
        config = TimeMoeConfig(num_classes=10, classifier_dropout=0.3)

        self.assertEqual(config.num_classes, 10)
        self.assertAlmostEqual(config.classifier_dropout, 0.3)

    def test_config_serialization(self):
        """Test that config can be converted to dict and back."""
        original_config = TimeMoeConfig(
            input_size=5, hidden_size=256, horizon_lengths=[1, 2], num_classes=3
        )

        config_dict = original_config.to_dict()

        # Check that important fields are in the dict
        self.assertEqual(config_dict["input_size"], 5)
        self.assertEqual(config_dict["hidden_size"], 256)
        self.assertEqual(config_dict["horizon_lengths"], [1, 2])
        self.assertEqual(config_dict["num_classes"], 3)
        self.assertEqual(config_dict["model_type"], "time_moe")


if __name__ == "__main__":
    unittest.main()
