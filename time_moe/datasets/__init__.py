#!/usr/bin/env python
from .binary_dataset import BinaryDataset
from .general_dataset import GeneralDataset

# Import classification data generators (optional dependencies)
try:
    from .classification_dataset import (
        create_sequence_classification_data,
        create_time_series_classification_data,
        create_token_classification_data,
        create_token_classification_regime_data,
        generate_sample_datasets,
        save_classification_data,
    )

    __all__ = [
        "BinaryDataset",
        "GeneralDataset",
        "create_sequence_classification_data",
        "create_token_classification_data",
        "create_time_series_classification_data",
        "create_token_classification_regime_data",
        "save_classification_data",
        "generate_sample_datasets",
    ]
except ImportError:
    # Classification data generators require numpy and scikit-learn
    __all__ = ["BinaryDataset", "GeneralDataset"]
