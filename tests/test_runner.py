#!/usr/bin/env python3
"""
Unit tests for Time-MoE runner module.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from time_moe.runner import TimeMoeRunner
    from time_moe.models.configuration_time_moe import TimeMoeConfig

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestTimeMoeRunner(unittest.TestCase):
    """Test TimeMoeRunner functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_output")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runner_initialization(self):
        """Test TimeMoeRunner initialization."""
        runner = TimeMoeRunner(model_path=None, output_path=self.output_path, seed=42)

        self.assertIsNone(runner.model_path)
        self.assertEqual(runner.output_path, self.output_path)
        self.assertEqual(runner.seed, 42)

    def test_load_model_from_scratch(self):
        """Test loading model from scratch with config."""
        # Create a test config file
        config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            horizon_lengths=[1],
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        )

        config_path = os.path.join(self.temp_dir, "config")
        os.makedirs(config_path, exist_ok=True)
        config.save_pretrained(config_path)

        runner = TimeMoeRunner(model_path=config_path)

        # Test forecasting model
        model = runner.load_model(
            from_scratch=True, task_type="forecasting", attn_implementation="eager"
        )

        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, "TimeMoeForPrediction")

    def test_load_model_classification_from_scratch(self):
        """Test loading classification model from scratch."""
        # Create a test config file
        config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            horizon_lengths=[1],
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        )

        config_path = os.path.join(self.temp_dir, "config")
        os.makedirs(config_path, exist_ok=True)
        config.save_pretrained(config_path)

        runner = TimeMoeRunner(model_path=config_path)

        # Test sequence classification model
        model = runner.load_model(
            from_scratch=True,
            task_type="sequence_classification",
            num_classes=5,
            classifier_dropout=0.2,
            attn_implementation="eager",
        )

        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, "TimeMoeForSequenceClassification")
        self.assertEqual(model.config.num_classes, 5)
        self.assertAlmostEqual(model.config.classifier_dropout, 0.2)

    def test_load_model_token_classification_from_scratch(self):
        """Test loading token classification model from scratch."""
        # Create a test config file
        config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            horizon_lengths=[1],
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        )

        config_path = os.path.join(self.temp_dir, "config")
        os.makedirs(config_path, exist_ok=True)
        config.save_pretrained(config_path)

        runner = TimeMoeRunner(model_path=config_path)

        # Test token classification model
        model = runner.load_model(
            from_scratch=True,
            task_type="token_classification",
            num_classes=3,
            attn_implementation="eager",
        )

        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, "TimeMoeForTokenClassification")
        self.assertEqual(model.config.num_classes, 3)

    def test_attention_implementation_selection(self):
        """Test attention implementation selection logic."""
        config = TimeMoeConfig(
            input_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            horizon_lengths=[1],
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        )

        config_path = os.path.join(self.temp_dir, "config")
        os.makedirs(config_path, exist_ok=True)
        config.save_pretrained(config_path)

        runner = TimeMoeRunner(model_path=config_path)

        # Test with eager attention (should work)
        model = runner.load_model(from_scratch=True, attn_implementation="eager")
        self.assertIsNotNone(model)

        # Test with auto (should fallback to eager)
        model = runner.load_model(from_scratch=True, attn_implementation="auto")
        self.assertIsNotNone(model)

    def test_train_model_batch_size_calculation(self):
        """Test batch size calculation logic in train_model."""
        runner = TimeMoeRunner()

        # Mock get_world_size to return 2
        import time_moe.utils.dist_util

        original_get_world_size = time_moe.utils.dist_util.get_world_size
        time_moe.utils.dist_util.get_world_size = lambda: 2

        try:
            # Test with global_batch_size only - should fail due to missing model_path
            with self.assertRaises(ValueError, msg="Model path is None"):
                runner.train_model(global_batch_size=16, normalization_method="min_max")

            # Test with micro_batch_size only
            with self.assertRaises(ValueError):  # Will fail due to missing model_path
                runner.train_model(micro_batch_size=4, normalization_method="min_max")

            # Test with neither (should raise ValueError)
            with self.assertRaises(ValueError):
                runner.train_model()

        finally:
            # Restore original function
            time_moe.utils.dist_util.get_world_size = original_get_world_size

    def test_runner_default_values(self):
        """Test runner default initialization values."""
        runner = TimeMoeRunner()

        self.assertIsNone(runner.model_path)
        self.assertEqual(runner.output_path, "logs/time_moe")
        self.assertEqual(runner.seed, 9899)


if __name__ == "__main__":
    unittest.main()
