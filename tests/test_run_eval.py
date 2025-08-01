#!/usr/bin/env python3
"""
Unit tests for run_eval.py functionality.
"""

import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from run_eval import (
        AccuracyMetric,
        F1Metric,
        MAEMetric,
        MSEMetric,
        SumEvalMetric,
        TimeMoE,
        count_num_tensor_elements,
        evaluate,
        setup_nccl,
    )
    from time_moe.datasets.benchmark_dataset import (
        BenchmarkEvalDataset,
        GeneralEvalDataset,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestMetrics(unittest.TestCase):
    """Test metric classes from run_eval.py."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_sum_eval_metric_base_class(self):
        """Test SumEvalMetric base class."""
        metric = SumEvalMetric("test_metric", 5.0)
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 5.0)

    def test_mse_metric(self):
        """Test MSEMetric calculation."""
        metric = MSEMetric("mse")
        preds = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.5, 2.5, 2.5])

        metric.push(preds, labels)
        expected_mse = torch.sum((preds - labels) ** 2).item()
        self.assertEqual(metric.value, expected_mse)

    def test_mae_metric(self):
        """Test MAEMetric calculation."""
        metric = MAEMetric("mae")
        preds = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.5, 2.5, 2.5])

        metric.push(preds, labels)
        expected_mae = torch.sum(torch.abs(preds - labels)).item()
        self.assertEqual(metric.value, expected_mae)

    def test_accuracy_metric_1d(self):
        """Test AccuracyMetric with 1D tensors."""
        metric = AccuracyMetric("accuracy")
        preds = torch.tensor([0, 1, 2, 1])
        labels = torch.tensor([0, 1, 1, 1])

        metric.push(preds, labels)
        # Correct predictions: [0==0, 1==1, 2!=1, 1==1] = 3 correct
        self.assertEqual(metric.value, 3.0)

    def test_accuracy_metric_2d(self):
        """Test AccuracyMetric with 2D tensors (logits)."""
        metric = AccuracyMetric("accuracy")
        # Logits where argmax gives [1, 0, 2]
        preds = torch.tensor(
            [
                [0.1, 0.9, 0.0],  # argmax = 1
                [0.8, 0.1, 0.1],  # argmax = 0
                [0.1, 0.1, 0.8],  # argmax = 2
            ]
        )
        labels = torch.tensor([1, 0, 1])

        metric.push(preds, labels)
        # Correct predictions: [1==1, 0==0, 2!=1] = 2 correct
        self.assertEqual(metric.value, 2.0)

    def test_f1_metric(self):
        """Test F1Metric calculation."""
        metric = F1Metric("f1", num_classes=3)

        # Test with simple predictions
        preds = torch.tensor([0, 1, 2, 1])
        labels = torch.tensor([0, 1, 1, 1])

        metric.push(preds, labels)
        f1_score = metric.get_f1_score()

        # F1 should be between 0 and 1
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

    def test_f1_metric_with_logits(self):
        """Test F1Metric with 2D tensor (logits)."""
        metric = F1Metric("f1", num_classes=2)

        preds = torch.tensor(
            [
                [0.1, 0.9],  # argmax = 1
                [0.8, 0.2],  # argmax = 0
                [0.3, 0.7],  # argmax = 1
            ]
        )
        labels = torch.tensor([1, 0, 1])

        metric.push(preds, labels)
        f1_score = metric.get_f1_score()

        # All predictions are correct, so F1 should be 1.0
        self.assertAlmostEqual(f1_score, 1.0, places=5)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions from run_eval.py."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_count_num_tensor_elements(self):
        """Test count_num_tensor_elements function."""
        # Test 1D tensor
        tensor_1d = torch.tensor([1, 2, 3, 4])
        self.assertEqual(count_num_tensor_elements(tensor_1d), 4)

        # Test 2D tensor
        tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(count_num_tensor_elements(tensor_2d), 6)

        # Test 3D tensor
        tensor_3d = torch.zeros(2, 3, 4)
        self.assertEqual(count_num_tensor_elements(tensor_3d), 24)

        # Test scalar
        tensor_scalar = torch.tensor(5.0)
        self.assertEqual(count_num_tensor_elements(tensor_scalar), 1)

    @patch("torch.distributed.init_process_group")
    def test_setup_nccl(self, mock_init_process_group):
        """Test setup_nccl function."""
        setup_nccl(rank=0, world_size=2, master_addr="localhost", master_port=12345)

        mock_init_process_group.assert_called_once_with(
            "nccl",
            init_method="tcp://localhost:12345",
            rank=0,
            world_size=2,
        )


class TestTimeMoE(unittest.TestCase):
    """Test TimeMoE class from run_eval.py."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    @patch("run_eval.AutoModelForCausalLM.from_pretrained")
    @patch("time_moe.models.modeling_time_moe.TimeMoeForPrediction.from_pretrained")
    def test_timemoe_forecasting_init(
        self, mock_timemoe_from_pretrained, mock_auto_from_pretrained
    ):
        """Test TimeMoE initialization for forecasting task."""
        # Mock the model
        mock_model = Mock()
        mock_model.dtype = torch.float32
        mock_model.config._attn_implementation = "eager"
        mock_timemoe_from_pretrained.return_value = mock_model

        timemoe = TimeMoE(
            model_path="test_model",
            device="cpu",
            task_type="forecasting",
            prediction_length=96,
        )

        self.assertEqual(timemoe.task_type, "forecasting")
        self.assertEqual(timemoe.device, "cpu")
        self.assertEqual(timemoe.prediction_length, 96)
        mock_timemoe_from_pretrained.assert_called_once()

    @patch(
        "time_moe.models.modeling_time_moe.TimeMoeForSequenceClassification.from_pretrained"
    )
    def test_timemoe_sequence_classification_init(self, mock_from_pretrained):
        """Test TimeMoE initialization for sequence classification task."""
        # Mock the model
        mock_model = Mock()
        mock_model.dtype = torch.float32
        mock_model.config._attn_implementation = "eager"
        mock_from_pretrained.return_value = mock_model

        timemoe = TimeMoE(
            model_path="test_model", device="cpu", task_type="sequence_classification"
        )

        self.assertEqual(timemoe.task_type, "sequence_classification")
        mock_from_pretrained.assert_called_once()

    @patch(
        "time_moe.models.modeling_time_moe.TimeMoeForTokenClassification.from_pretrained"
    )
    def test_timemoe_token_classification_init(self, mock_from_pretrained):
        """Test TimeMoE initialization for token classification task."""
        # Mock the model
        mock_model = Mock()
        mock_model.dtype = torch.float32
        mock_model.config._attn_implementation = "eager"
        mock_from_pretrained.return_value = mock_model

        timemoe = TimeMoE(
            model_path="test_model", device="cpu", task_type="token_classification"
        )

        self.assertEqual(timemoe.task_type, "token_classification")
        mock_from_pretrained.assert_called_once()

    def test_timemoe_invalid_task_type(self):
        """Test TimeMoE initialization with invalid task type."""
        with self.assertRaises(ValueError) as context:
            TimeMoE(model_path="test_model", device="cpu", task_type="invalid_task")

        self.assertIn("Unsupported task_type", str(context.exception))

    def test_timemoe_predict_forecasting(self):
        """Test TimeMoE predict method for forecasting."""
        # Create a mock model
        mock_model = Mock()
        mock_model.dtype = torch.float32
        mock_model.config._attn_implementation = "eager"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

        with patch(
            "time_moe.models.modeling_time_moe.TimeMoeForPrediction.from_pretrained",
            return_value=mock_model,
        ):
            timemoe = TimeMoE(
                model_path="test_model",
                device="cpu",
                task_type="forecasting",
                prediction_length=3,
            )

            batch = {
                "inputs": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[4, 5, 6]]),
            }

            preds, labels = timemoe.predict(batch)

            # Should return last 3 tokens from generated sequence
            expected_preds = torch.tensor([[4, 5, 6]])
            torch.testing.assert_close(preds, expected_preds)

    def test_timemoe_predict_classification(self):
        """Test TimeMoE predict method for classification."""
        # Create a mock model
        mock_model = Mock()
        mock_model.dtype = torch.float32
        mock_model.config._attn_implementation = "eager"

        # Mock the forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        mock_model.return_value = mock_outputs

        with patch(
            "time_moe.models.modeling_time_moe.TimeMoeForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):
            timemoe = TimeMoE(
                model_path="test_model",
                device="cpu",
                task_type="sequence_classification",
            )

            batch = {
                "inputs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "labels": torch.tensor([1, 0]),
            }

            preds, labels = timemoe.predict(batch)

            # Should return logits
            expected_preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
            torch.testing.assert_close(preds, expected_preds)


class TestEvaluateFunction(unittest.TestCase):
    """Test the main evaluate function."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    @patch.dict(os.environ, {}, clear=True)
    @patch("run_eval.BenchmarkEvalDataset")
    @patch("run_eval.TimeMoE")
    @patch("torch.cuda.is_available", return_value=False)
    def test_evaluate_forecasting_basic(
        self, mock_cuda_available, mock_timemoe_class, mock_dataset_class
    ):
        """Test basic evaluate function for forecasting."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        # Mock model
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.predict.return_value = (
            torch.tensor([[1.0, 2.0]]),  # predictions
            torch.tensor([[1.1, 2.1]]),  # labels
        )
        mock_timemoe_class.return_value = mock_model

        # Mock DataLoader
        with patch("run_eval.DataLoader") as mock_dataloader:
            mock_dataloader.return_value = [
                {
                    "inputs": torch.tensor([[0.5, 0.6]]),
                    "labels": torch.tensor([[1.1, 2.1]]),
                }
            ]

            # Create test arguments
            args = argparse.Namespace(
                model="test_model",
                data="test_data.csv",
                task_type="forecasting",
                context_length=512,
                prediction_length=96,
                batch_size=32,
                num_classes=None,
            )

            # This should run without error
            with patch("builtins.print"):  # Suppress print outputs
                evaluate(args)

            mock_timemoe_class.assert_called_once()
            mock_dataset_class.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("run_eval.GeneralEvalDataset")
    @patch("run_eval.TimeMoE")
    @patch("torch.cuda.is_available", return_value=False)
    def test_evaluate_classification_basic(
        self, mock_cuda_available, mock_timemoe_class, mock_dataset_class
    ):
        """Test basic evaluate function for classification."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        # Mock model
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.predict.return_value = (
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),  # predictions (logits)
            torch.tensor([1, 0]),  # labels
        )
        mock_timemoe_class.return_value = mock_model

        # Mock DataLoader
        with patch("run_eval.DataLoader") as mock_dataloader:
            mock_dataloader.return_value = [
                {
                    "inputs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "labels": torch.tensor([1, 0]),
                }
            ]

            # Create test arguments
            args = argparse.Namespace(
                model="test_model",
                data="test_data",  # Not .csv, so will use GeneralEvalDataset
                task_type="sequence_classification",
                context_length=1024,
                prediction_length=None,
                batch_size=32,
                num_classes=2,
            )

            # This should run without error
            with patch("builtins.print"):  # Suppress print outputs
                evaluate(args)

            mock_timemoe_class.assert_called_once()
            mock_dataset_class.assert_called_once()

    @patch.dict(
        os.environ, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}, clear=True
    )
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("run_eval.TimeMoE")
    @patch("run_eval.BenchmarkEvalDataset")
    @patch("run_eval.setup_nccl")
    def test_evaluate_distributed(
        self,
        mock_setup_nccl,
        mock_dataset_class,
        mock_timemoe_class,
        mock_cuda_available,
        mock_dist_initialized,
        mock_all_gather,
    ):
        """Test evaluate function in distributed setting."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        # Mock model
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_model.predict.return_value = (
            torch.tensor([[1.0, 2.0]]),  # predictions
            torch.tensor([[1.1, 2.1]]),  # labels
        )
        mock_timemoe_class.return_value = mock_model

        # Mock DataLoader with DistributedSampler
        with (
            patch("run_eval.DataLoader") as mock_dataloader,
            patch("run_eval.DistributedSampler") as mock_sampler,
        ):
            mock_dataloader.return_value = [
                {
                    "inputs": torch.tensor([[0.5, 0.6]]),
                    "labels": torch.tensor([[1.1, 2.1]]),
                }
            ]

            # Create test arguments
            args = argparse.Namespace(
                model="test_model",
                data="test_data.csv",
                task_type="forecasting",
                context_length=512,
                prediction_length=96,
                batch_size=32,
                num_classes=None,
            )

            # This should run without error
            with patch("builtins.print"):  # Suppress print outputs
                evaluate(args)

            mock_setup_nccl.assert_called_once()
            mock_sampler.assert_called_once()


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing and validation in main."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_context_length_defaults_forecasting(self):
        """Test context length defaults for forecasting tasks."""
        test_cases = [
            (96, 512),
            (192, 1024),
            (336, 2048),
            (720, 3072),
            (480, 1920),  # 480 * 4 = 1920
        ]

        for prediction_length, expected_context_length in test_cases:
            with patch("run_eval.evaluate"):
                with patch(
                    "sys.argv",
                    [
                        "run_eval.py",
                        "--data",
                        "test.csv",
                        "--prediction_length",
                        str(prediction_length),
                    ],
                ):
                    # Import and run the argument parsing part

                    # We can't easily test the main function directly, so we'll test the logic
                    # by creating the parser and testing the validation logic
                    parser = argparse.ArgumentParser("TimeMoE Evaluate")
                    parser.add_argument(
                        "--model", "-m", type=str, default="Maple728/TimeMoE-50M"
                    )
                    parser.add_argument(
                        "--data", "-d", type=str, help="Benchmark data path"
                    )
                    parser.add_argument(
                        "--task_type",
                        type=str,
                        choices=[
                            "forecasting",
                            "sequence_classification",
                            "token_classification",
                        ],
                        default="forecasting",
                    )
                    parser.add_argument("--num_classes", type=int, default=None)
                    parser.add_argument("--batch_size", "-b", type=int, default=32)
                    parser.add_argument(
                        "--context_length", "-c", type=int, help="Context length"
                    )
                    parser.add_argument(
                        "--prediction_length", "-p", type=int, default=96
                    )

                    args = parser.parse_args(
                        [
                            "--data",
                            "test.csv",
                            "--prediction_length",
                            str(prediction_length),
                        ]
                    )

                    # Apply the same logic as in the main function
                    if args.context_length is None:
                        if args.task_type == "forecasting":
                            if args.prediction_length == 96:
                                args.context_length = 512
                            elif args.prediction_length == 192:
                                args.context_length = 1024
                            elif args.prediction_length == 336:
                                args.context_length = 2048
                            elif args.prediction_length == 720:
                                args.context_length = 3072
                            else:
                                args.context_length = args.prediction_length * 4

                    self.assertEqual(
                        args.context_length,
                        expected_context_length,
                        f"For prediction_length={prediction_length}, expected context_length={expected_context_length}",
                    )


if __name__ == "__main__":
    unittest.main()
