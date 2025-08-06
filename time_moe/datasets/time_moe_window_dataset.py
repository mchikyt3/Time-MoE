#!/usr/bin/env python
import random

import numpy as np
import torch

from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoEWindowDataset:
    """
    A dataset class for generating non-overlapping sliding windows from a time series dataset.
    This is useful for training models that require fixed-length input sequences and corresponding labels.

    Attributes
    ----------
        dataset (TimeSeriesDataset): The underlying time series dataset.
        context_length (int): Length of the input context window.
        prediction_length (int): Length of the prediction window. Defaults to 0.
        window_size (int): Total size of the sliding window (context_length + prediction_length).
        window_size_plus_one (int): Total size of the sliding window plus one.
        stride (int): Step size for sliding the window. Defaults to window_size.
        sub_seq_indexes (list): List of tuples containing sequence indices and their corresponding offsets.

    Methods
    -------
        __len__():
            Returns the total number of sliding windows in the dataset.
        __iter__():
            Iterates over the dataset, yielding one sliding window at a time.
        __getitem__(seq_idx):
            Retrieves a sliding window, its labels, and a loss mask.

    Example:
        >>> dataset = TimeSeriesDataset(...)  # Assume this is a predefined dataset
        >>> context_length = 10
        >>> prediction_length = 5
        >>> window_dataset = TimeMoEWindowDataset(
        ...     dataset, context_length, prediction_length
        ... )
        >>> for sample in window_dataset:
        >>>     print(sample['input_ids'], sample['labels'], sample['loss_masks'])
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        stride: int = None,
        task_type: str = "forecasting",
        num_classes: int = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size
        
        # Classification-specific attributes
        self.task_type = task_type
        self.num_classes = num_classes
        self.is_classification = task_type in ["sequence_classification", "token_classification"]

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
            
        self.sub_seq_indexes = []
        
        if self.is_classification:
            # For classification, use each sequence as-is (no sliding windows)
            for seq_idx in iterator:
                self.sub_seq_indexes.append((seq_idx, 0))
        else:
            # For forecasting, use sliding windows (original behavior)
            for seq_idx in iterator:
                n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
                # Skip sequences with fewer than 2 points
                if n_points < 2:
                    continue
                self.sub_seq_indexes.append((seq_idx, 0))
                for offset_idx in range(
                    self.stride, n_points - self.window_size_plus_one + 1, self.stride
                ):
                    self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        
        if self.is_classification:
            # For classification, we need the full dict with labels
            # Access the raw data directly to get full dict with labels
            if hasattr(self.dataset, 'datasets') and self.dataset.datasets:
                # For TimeMoEDataset, get from the underlying dataset
                dataset_idx = self._find_dataset_idx(seq_i)
                dataset_offset = seq_i - self.dataset.cumsum_lengths[dataset_idx]
                # Use get_raw_item if available, otherwise fallback to data access
                if hasattr(self.dataset.datasets[dataset_idx], 'get_raw_item'):
                    raw_item = self.dataset.datasets[dataset_idx].get_raw_item(dataset_offset)
                elif hasattr(self.dataset.datasets[dataset_idx], 'data'):
                    raw_item = self.dataset.datasets[dataset_idx].data[dataset_offset]
                else:
                    raw_item = self.dataset.datasets[dataset_idx][dataset_offset]
            else:
                # For single dataset case
                if hasattr(self.dataset, 'get_raw_item'):
                    raw_item = self.dataset.get_raw_item(seq_i)
                else:
                    raw_item = self.dataset[seq_i]
            
            # Handle classification data format
            return self._process_classification_item(raw_item, offset_i)
        else:
            # For forecasting, get processed sequence (original behavior)
            raw_item = self.dataset[seq_i]
            return self._process_forecasting_item(raw_item, offset_i)
    
    def _find_dataset_idx(self, seq_i):
        """Find which dataset contains the sequence index."""
        for i, cumsum in enumerate(self.dataset.cumsum_lengths[1:], 1):
            if seq_i < cumsum:
                return i - 1
        return len(self.dataset.cumsum_lengths) - 2

    def _process_forecasting_item(self, raw_item, offset_i):
        """Process item for forecasting task."""
        # Extract sequence from raw_item (could be list or dict)
        if isinstance(raw_item, dict):
            seq = raw_item.get("sequence", raw_item)
        else:
            seq = raw_item
            
        # Get the window
        seq = seq[offset_i : offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), "constant", constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), "constant", constant_values=0)

        return {"input_ids": seq[:-1], "labels": seq[1:], "loss_masks": loss_mask}

    def _process_classification_item(self, raw_item, offset_i):
        """Process item for classification task."""
        if not isinstance(raw_item, dict):
            raise ValueError(
                f"Classification data must be in dict format with 'sequence' and 'label' keys, "
                f"got {type(raw_item)}"
            )
        
        sequence = raw_item.get("sequence")
        label = raw_item.get("label")
        
        if sequence is None:
            raise ValueError("Classification data must have 'sequence' field")
        if label is None:
            raise ValueError("Classification data must have 'label' field")

        # For classification, we use the full sequence length as context
        # Truncate or pad sequence to context_length
        if len(sequence) > self.context_length:
            sequence = sequence[:self.context_length]
        elif len(sequence) < self.context_length:
            # Pad with zeros
            sequence = sequence + [0.0] * (self.context_length - len(sequence))

        # Handle labels based on task type
        if self.task_type == "sequence_classification":
            # Label should be a single integer
            if isinstance(label, (list, np.ndarray)):
                raise ValueError(
                    f"For sequence classification, label should be a single integer, "
                    f"got {type(label)}"
                )
            processed_label = int(label)
            
        elif self.task_type == "token_classification":
            # Label should be a list/array of integers matching sequence length
            if not isinstance(label, (list, np.ndarray)):
                raise ValueError(
                    f"For token classification, label should be a list/array, "
                    f"got {type(label)}"
                )

            label_list = list(label)

            # Truncate or pad labels to match sequence length
            if len(label_list) > self.context_length:
                label_list = label_list[:self.context_length]
            elif len(label_list) < self.context_length:
                # Pad with -100 (ignore index for loss calculation)
                label_list = label_list + [-100] * (
                    self.context_length - len(label_list)
                )

            processed_label = label_list

        # Convert to tensors
        import torch
        input_ids = torch.tensor(sequence, dtype=torch.float32)
        
        if self.task_type == "sequence_classification":
            labels = torch.tensor(processed_label, dtype=torch.long)
        else:  # token_classification
            labels = torch.tensor(processed_label, dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


class UniversalTimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        shuffle: bool = False,
        **kwrags,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][
                start_idx_in_seq : start_idx_in_seq + offset
            ]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
        }
