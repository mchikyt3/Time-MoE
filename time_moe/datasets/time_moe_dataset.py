#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os

import numpy as np
import torch

from .binary_dataset import BinaryDataset
from .general_dataset import GeneralDataset
from .ts_dataset import TimeSeriesDataset


class TimeMoEDataset(TimeSeriesDataset):
    def __init__(self, data_folder, normalization_method=None):
        self.data_folder = data_folder
        self.normalization_method = normalization_method
        self.datasets = []
        self.num_tokens = None

        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == "max":
                self.normalization_method = max_scaler
            elif normalization_method.lower() == "zero":
                self.normalization_method = zero_scaler
            else:
                raise ValueError(
                    f"Unknown normalization method: {normalization_method}"
                )
        else:
            self.normalization_method = normalization_method

        if BinaryDataset.is_valid_path(self.data_folder):
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            ds = GeneralDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    fn_path = os.path.join(root, file)
                    if (
                        file != BinaryDataset.meta_file_name
                        and GeneralDataset.is_valid_path(fn_path)
                    ):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(self.cumsum_lengths[-1] + len(ds))
        self.num_sequences = self.cumsum_lengths[-1]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(
                f"Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}"
            )
        elif seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        seq = self.datasets[dataset_idx][dataset_offset]

        if self.normalization_method is not None:
            seq = self.normalization_method(seq)
        return seq

    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(
                f"Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}"
            )
        elif seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])

        return self.num_tokens


def zero_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    # std_val = seq.std(dtype=np.float64)
    # mean_val = seq.mean(dtype=np.float64)
    mean_val = seq.mean()
    std_val = seq.std()
    if std_val == 0:
        normed_seq = seq - mean_val
    else:
        normed_seq = (seq - mean_val) / std_val

    return normed_seq.astype(origin_dtype)


def max_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    # max_val = np.abs(seq).max(dtype=np.float64)
    max_val = np.abs(seq).max()
    if max_val == 0:
        normed_seq = seq
    else:
        normed_seq = seq / max_val

    return normed_seq.astype(origin_dtype)


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index


class TimeMoeClassificationDataset:
    """
    A dataset class for time series classification tasks.
    Supports both sequence classification (one label per sequence) and
    token classification (one label per timestep).
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        task_type: str = "sequence_classification",
        num_classes: int = None,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.task_type = task_type
        self.num_classes = num_classes

        # Validate task type
        if task_type not in ["sequence_classification", "token_classification"]:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Must be 'sequence_classification' or 'token_classification'"
            )

        # Process data to extract sequences and labels
        self.sequences = []
        self.labels = []

        for i in range(len(dataset)):
            item = dataset.data[i] if hasattr(dataset, "data") else dataset[i]

            if isinstance(item, dict):
                sequence = item.get("sequence", item)
                label = item.get("label", None)

                if label is None:
                    raise ValueError(
                        f"No label found for item {i}. Classification data must have 'label' field."
                    )

                # Truncate or pad sequence to context_length
                if len(sequence) > context_length:
                    sequence = sequence[:context_length]
                elif len(sequence) < context_length:
                    # Pad with zeros
                    sequence = sequence + [0.0] * (context_length - len(sequence))

                # Handle labels based on task type
                if task_type == "sequence_classification":
                    # Label should be a single integer
                    if isinstance(label, (list, np.ndarray)):
                        raise ValueError(
                            f"For sequence classification, label should be a single integer, got {type(label)}"
                        )
                    processed_label = int(label)

                elif task_type == "token_classification":
                    # Label should be a list/array of integers matching sequence length
                    if not isinstance(label, (list, np.ndarray)):
                        raise ValueError(
                            f"For token classification, label should be a list/array, got {type(label)}"
                        )

                    label_list = list(label)

                    # Truncate or pad labels to match sequence length
                    if len(label_list) > context_length:
                        label_list = label_list[:context_length]
                    elif len(label_list) < context_length:
                        # Pad with -100 (ignore index for loss calculation)
                        label_list = label_list + [-100] * (
                            context_length - len(label_list)
                        )

                    processed_label = label_list

                self.sequences.append(sequence)
                self.labels.append(processed_label)
            else:
                raise ValueError(
                    f"Item {i} is not a dictionary. Classification data must be in dict format with 'sequence' and 'label' keys."
                )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Convert to tensors
        input_ids = torch.tensor(sequence, dtype=torch.float32)

        if self.task_type == "sequence_classification":
            labels = torch.tensor(label, dtype=torch.long)
        else:  # token_classification
            labels = torch.tensor(label, dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


class TimeMoeClassificationWrapper:
    """
    Wrapper to make classification dataset compatible with the existing training infrastructure.
    """

    def __init__(
        self,
        data_path: str,
        context_length: int,
        task_type: str = "sequence_classification",
        num_classes: int = None,
        normalization_method=None,
    ):
        # Load the base dataset
        base_dataset = TimeMoEDataset(
            data_path, normalization_method=normalization_method
        )

        # Create classification dataset
        self.classification_dataset = TimeMoeClassificationDataset(
            dataset=base_dataset.datasets[0],  # Assuming single dataset for now
            context_length=context_length,
            task_type=task_type,
            num_classes=num_classes,
        )

    def __len__(self):
        return len(self.classification_dataset)

    def __getitem__(self, idx):
        return self.classification_dataset[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
