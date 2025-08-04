#!/usr/bin/env python3
"""
Unit tests for Time-MoE utility modules.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from time_moe.utils.dist_util import get_world_size
    from time_moe.utils.log_util import get_logger, is_local_rank_0, log_in_local_rank_0

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestLogUtil(unittest.TestCase):
    """Test logging utilities."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_get_logger(self):
        """Test logger creation and configuration."""
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")

        # Test with custom level
        logger_debug = get_logger("test_debug", level="DEBUG")
        self.assertEqual(logger_debug.level, 10)  # DEBUG level

        # Test caching
        logger_cached = get_logger("test_logger")
        self.assertIs(logger, logger_cached)

        # Test update
        logger_updated = get_logger("test_logger", level="ERROR", update=True)
        self.assertEqual(logger_updated.level, 40)  # ERROR level

    def test_is_local_rank_0(self):
        """Test local rank detection."""
        # Save original LOCAL_RANK
        original_local_rank = os.environ.get("LOCAL_RANK")

        try:
            # Test with no LOCAL_RANK set
            if "LOCAL_RANK" in os.environ:
                del os.environ["LOCAL_RANK"]
            self.assertTrue(is_local_rank_0())

            # Test with LOCAL_RANK=0
            os.environ["LOCAL_RANK"] = "0"
            self.assertTrue(is_local_rank_0())

            # Test with LOCAL_RANK=1
            os.environ["LOCAL_RANK"] = "1"
            self.assertFalse(is_local_rank_0())

        finally:
            # Restore original LOCAL_RANK
            if original_local_rank is not None:
                os.environ["LOCAL_RANK"] = original_local_rank
            elif "LOCAL_RANK" in os.environ:
                del os.environ["LOCAL_RANK"]

    def test_log_in_local_rank_0(self):
        """Test logging only in local rank 0."""
        # Save original LOCAL_RANK
        original_local_rank = os.environ.get("LOCAL_RANK")

        try:
            # Set LOCAL_RANK to 0 (should log)
            os.environ["LOCAL_RANK"] = "0"

            # Create a test logger
            test_logger = get_logger("test_log_function")

            # These should not raise exceptions
            log_in_local_rank_0("Test info message", used_logger=test_logger)
            log_in_local_rank_0("Test warning", type="warn", used_logger=test_logger)
            log_in_local_rank_0("Test error", type="error", used_logger=test_logger)

            # Test with non-rank-0 (should still not raise, just not log)
            os.environ["LOCAL_RANK"] = "1"
            log_in_local_rank_0("Should not log", used_logger=test_logger)

        finally:
            # Restore original LOCAL_RANK
            if original_local_rank is not None:
                os.environ["LOCAL_RANK"] = original_local_rank
            elif "LOCAL_RANK" in os.environ:
                del os.environ["LOCAL_RANK"]


class TestDistUtil(unittest.TestCase):
    """Test distributed training utilities."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")

    def test_get_world_size(self):
        """Test getting world size."""
        # Save original environment variables
        original_world_size = os.environ.get("WORLD_SIZE")

        try:
            # Test with no WORLD_SIZE set (should default to 1)
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]
            world_size = get_world_size()
            self.assertEqual(world_size, 1)

            # Test with WORLD_SIZE set
            os.environ["WORLD_SIZE"] = "4"
            world_size = get_world_size()
            self.assertEqual(world_size, 4)

        finally:
            # Restore original WORLD_SIZE
            if original_world_size is not None:
                os.environ["WORLD_SIZE"] = original_world_size
            elif "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]


if __name__ == "__main__":
    unittest.main()
