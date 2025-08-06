#!/usr/bin/env python3
"""
Unit tests for Time-MoE model components.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from torch import nn

    from time_moe.models.configuration_time_moe import TimeMoeConfig
    from time_moe.models.modeling_time_moe import (
        TimeMoeAttention,
        TimeMoeClassificationHead,
        TimeMoeDecoderLayer,
        TimeMoeForPrediction,
        TimeMoeForSequenceClassification,
        TimeMoeForTokenClassification,
        TimeMoeInputEmbedding,
        TimeMoeMLP,
        TimeMoeModel,
        TimeMoeRMSNorm,
        load_balancing_loss_func,
        repeat_kv,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestTimeMoeComponents(unittest.TestCase):
    """Test individual Time-MoE model components."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        # Create minimal config for testing
        self.config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            horizon_lengths=[1, 2],
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )

    def test_input_embedding(self):
        """Test TimeMoeInputEmbedding component."""
        embedding = TimeMoeInputEmbedding(self.config)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_data = torch.randn(batch_size, seq_len, self.config.input_size)

        output = embedding(input_data)

        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertEqual(output.dtype, torch.float32)

    def test_rms_norm(self):
        """Test TimeMoeRMSNorm component."""
        rms_norm = TimeMoeRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_data = torch.randn(batch_size, seq_len, self.config.hidden_size)

        output = rms_norm(input_data)

        self.assertEqual(output.shape, input_data.shape)
        # Test that the variance is approximately 1 (for RMS norm)
        variance = output.pow(2).mean(-1)
        # RMS norm should produce vectors with normalized variance
        self.assertTrue(torch.allclose(variance, torch.ones_like(variance), atol=1e-3))

    def test_mlp(self):
        """Test TimeMoeMLP component."""
        mlp = TimeMoeMLP(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_data = torch.randn(batch_size, seq_len, self.config.hidden_size)

        output, aux_loss = mlp(input_data)

        self.assertEqual(output.shape, input_data.shape)

    def test_attention(self):
        """Test TimeMoeAttention component."""
        attention = TimeMoeAttention(self.config, layer_idx=0)

        # Test forward pass
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

        outputs = attention(hidden_states)

        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        self.assertEqual(output.shape, hidden_states.shape)

    def test_decoder_layer(self):
        """Test TimeMoeDecoderLayer component."""
        decoder_layer = TimeMoeDecoderLayer(self.config, layer_idx=0)

        # Test forward pass
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)

        outputs = decoder_layer(hidden_states)

        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        self.assertEqual(output.shape, hidden_states.shape)

    def test_classification_head(self):
        """Test TimeMoeClassificationHead component."""
        config_with_classes = TimeMoeConfig(
            hidden_size=32, num_classes=5, classifier_dropout=0.1
        )

        classification_head = TimeMoeClassificationHead(
            hidden_size=config_with_classes.hidden_size,
            num_classes=config_with_classes.num_classes,
            dropout=config_with_classes.classifier_dropout,
        )

        # Test 1: Token classification (3D input)
        batch_size, seq_len = 2, 10
        hidden_states_3d = torch.randn(
            batch_size, seq_len, config_with_classes.hidden_size
        )

        output_3d = classification_head(hidden_states_3d)
        expected_shape_3d = (batch_size, seq_len, config_with_classes.num_classes)
        self.assertEqual(output_3d.shape, expected_shape_3d)

        # Test 2: Sequence classification (2D input - last token hidden states)
        hidden_states_2d = torch.randn(batch_size, config_with_classes.hidden_size)

        output_2d = classification_head(hidden_states_2d)
        expected_shape_2d = (batch_size, config_with_classes.num_classes)
        self.assertEqual(output_2d.shape, expected_shape_2d)


class TestTimeMoeModels(unittest.TestCase):
    """Test complete Time-MoE models."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        # Create minimal config for testing
        self.config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,  # Single layer for speed
            num_attention_heads=4,
            num_key_value_heads=2,
            horizon_lengths=[1, 2],
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )

    def test_timemoe_model(self):
        """Test TimeMoeModel (backbone)."""
        model = TimeMoeModel(self.config)

        # Test forward pass
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

        outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (batch_size, seq_len, self.config.hidden_size),
        )

    def test_timemoe_for_prediction(self):
        """Test TimeMoeForPrediction model."""
        model = TimeMoeForPrediction(self.config)

        # Test forward pass
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, self.config.input_size)

        outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs.logits)
        # Output should have shape [batch_size, seq_len, input_size * horizon_length]
        expected_shape = (
            batch_size,
            seq_len,
            self.config.input_size * self.config.horizon_lengths[0],
        )
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_timemoe_for_sequence_classification(self):
        """Test TimeMoeForSequenceClassification model."""
        config_with_classes = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            num_classes=3,
            classifier_dropout=0.1,
        )

        model = TimeMoeForSequenceClassification(config_with_classes)

        # Test forward pass
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, config_with_classes.input_size)

        outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(
            outputs.logits.shape, (batch_size, config_with_classes.num_classes)
        )

    def test_timemoe_for_token_classification(self):
        """Test TimeMoeForTokenClassification model."""
        config_with_classes = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            num_classes=5,
            classifier_dropout=0.1,
        )

        model = TimeMoeForTokenClassification(config_with_classes)

        # Test forward pass
        batch_size, seq_len = 2, 8
        input_ids = torch.randn(batch_size, seq_len, config_with_classes.input_size)

        outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(
            outputs.logits.shape, (batch_size, seq_len, config_with_classes.num_classes)
        )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_repeat_kv(self):
        """Test repeat_kv function."""
        batch_size, num_kv_heads, seq_len, head_dim = 2, 4, 8, 16
        hidden_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

        # Test with n_rep = 1 (should return unchanged)
        result = repeat_kv(hidden_states, 1)
        self.assertEqual(result.shape, hidden_states.shape)
        self.assertTrue(torch.equal(result, hidden_states))

        # Test with n_rep = 2
        result = repeat_kv(hidden_states, 2)
        expected_shape = (batch_size, num_kv_heads * 2, seq_len, head_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_load_balancing_loss_func(self):
        """Test load balancing loss function."""
        # Test with None gate_logits
        loss = load_balancing_loss_func(None, top_k=2, num_experts=4)
        self.assertEqual(loss, 0.0)

        # Test with empty gate_logits
        loss = load_balancing_loss_func([], top_k=2, num_experts=4)
        self.assertEqual(loss, 0.0)

        # Test with valid gate_logits
        gate_logits = [
            torch.randn(16, 4),  # [batch_size * seq_len, num_experts]
            torch.randn(16, 4),
        ]

        loss = load_balancing_loss_func(gate_logits, top_k=2, num_experts=4)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_model_training_mode(self):
        """Test that models can be set to training/evaluation mode."""
        config = TimeMoeConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_experts=2,
            num_experts_per_tok=1,
        )

        model = TimeMoeForPrediction(config)

        # Test training mode
        model.train()
        self.assertTrue(model.training)

        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)

    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_experts=2,
            num_experts_per_tok=1,
        )

        model = TimeMoeForPrediction(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(
            total_params, trainable_params
        )  # All params should be trainable by default


if __name__ == "__main__":
    unittest.main()
