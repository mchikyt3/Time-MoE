#!/usr/bin/env python3
"""
Unit tests for Time-MoE datasets.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from time_moe.datasets.general_dataset import GeneralDataset, read_file_by_extension
    from time_moe.datasets.binary_dataset import BinaryDataset

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestGeneralDataset(unittest.TestCase):
    """Test GeneralDataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        # Create temporary test data
        self.temp_dir = tempfile.mkdtemp()

        # Test data
        self.test_data = [
            [1, 2, 3, 4, 5],
            [6, 7, 8],
            [9, 10, 11, 12],
        ]

        self.test_data_with_dict = [
            {"sequence": [1, 2, 3, 4, 5], "metadata": "test1"},
            {"sequence": [6, 7, 8], "metadata": "test2"},
            {"sequence": [9, 10, 11, 12], "metadata": "test3"},
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_file_loading(self):
        """Test loading from JSON file."""
        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, "w") as f:
            json.dump(self.test_data, f)

        dataset = GeneralDataset(json_file)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], [1, 2, 3, 4, 5])
        self.assertEqual(dataset[1], [6, 7, 8])
        self.assertEqual(dataset[2], [9, 10, 11, 12])

    def test_jsonl_file_loading(self):
        """Test loading from JSONL file."""
        jsonl_file = os.path.join(self.temp_dir, "test.jsonl")
        with open(jsonl_file, "w") as f:
            for item in self.test_data:
                f.write(json.dumps(item) + "\n")

        dataset = GeneralDataset(jsonl_file)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], [1, 2, 3, 4, 5])

    def test_npy_file_loading(self):
        """Test loading from numpy file."""
        npy_file = os.path.join(self.temp_dir, "test.npy")
        # Use object dtype to handle heterogeneous array lengths
        np.save(npy_file, np.array(self.test_data, dtype=object))

        dataset = GeneralDataset(npy_file)
        self.assertEqual(len(dataset), 3)
        np.testing.assert_array_equal(dataset[0], [1, 2, 3, 4, 5])

    def test_dict_sequence_extraction(self):
        """Test extracting sequences from dict format."""
        json_file = os.path.join(self.temp_dir, "test_dict.json")
        with open(json_file, "w") as f:
            json.dump(self.test_data_with_dict, f)

        dataset = GeneralDataset(json_file)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], [1, 2, 3, 4, 5])
        self.assertEqual(dataset[1], [6, 7, 8])

    def test_get_num_tokens(self):
        """Test getting total number of tokens."""
        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, "w") as f:
            json.dump(self.test_data, f)

        dataset = GeneralDataset(json_file)
        expected_tokens = sum(len(seq) for seq in self.test_data)
        self.assertEqual(dataset.get_num_tokens(), expected_tokens)

        # Test caching
        self.assertEqual(dataset.get_num_tokens(), expected_tokens)

    def test_get_sequence_length_by_idx(self):
        """Test getting sequence length by index."""
        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, "w") as f:
            json.dump(self.test_data, f)

        dataset = GeneralDataset(json_file)
        self.assertEqual(dataset.get_sequence_length_by_idx(0), 5)
        self.assertEqual(dataset.get_sequence_length_by_idx(1), 3)
        self.assertEqual(dataset.get_sequence_length_by_idx(2), 4)

    def test_is_valid_path(self):
        """Test path validation."""
        # Create valid files
        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, "w") as f:
            json.dump([], f)

        npy_file = os.path.join(self.temp_dir, "test.npy")
        np.save(npy_file, [])

        # Test valid paths
        self.assertTrue(GeneralDataset.is_valid_path(json_file))
        self.assertTrue(GeneralDataset.is_valid_path(npy_file))

        # Test invalid paths
        self.assertFalse(GeneralDataset.is_valid_path("nonexistent.json"))

        txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("test")
        self.assertFalse(GeneralDataset.is_valid_path(txt_file))


class TestBinaryDataset(unittest.TestCase):
    """Test BinaryDataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_binary_dataset_creation(self):
        """Test creating binary dataset."""
        # Create a proper BinaryDataset structure
        dataset_dir = os.path.join(self.temp_dir, "binary_dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        # Create meta.json file
        meta_info = {
            "num_sequences": 2,
            "dtype": "float32",
            "scales": [{"offset": 0, "length": 5}, {"offset": 5, "length": 3}],
            "files": {"data-1-of-1.bin": 8},
        }

        meta_file = os.path.join(dataset_dir, "meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta_info, f)

        # Create binary data file
        binary_file = os.path.join(dataset_dir, "data-1-of-1.bin")
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        with open(binary_file, "wb") as f:
            f.write(test_data.tobytes())

        # Test binary dataset creation
        dataset = BinaryDataset(dataset_dir)
        self.assertEqual(len(dataset), 2)


class TestReadFileByExtension(unittest.TestCase):
    """Test read_file_by_extension utility function."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

        self.temp_dir = tempfile.mkdtemp()
        self.test_data = [1, 2, 3, 4, 5]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_json(self):
        """Test reading JSON files."""
        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, "w") as f:
            json.dump(self.test_data, f)

        data = read_file_by_extension(json_file)
        self.assertEqual(data, self.test_data)

    def test_read_npy(self):
        """Test reading numpy files."""
        npy_file = os.path.join(self.temp_dir, "test.npy")
        np.save(npy_file, self.test_data)

        data = read_file_by_extension(npy_file)
        np.testing.assert_array_equal(data, self.test_data)

    def test_read_jsonl(self):
        """Test reading JSONL files."""
        jsonl_file = os.path.join(self.temp_dir, "test.jsonl")
        lines = [[1, 2], [3, 4], [5, 6]]

        with open(jsonl_file, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        data = read_file_by_extension(jsonl_file)
        self.assertEqual(data, lines)

    def test_unsupported_extension(self):
        """Test handling unsupported file extensions."""
        txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("test content")

        with self.assertRaises(RuntimeError):
            read_file_by_extension(txt_file)


if __name__ == "__main__":
    unittest.main()
