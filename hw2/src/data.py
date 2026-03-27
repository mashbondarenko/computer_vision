from typing import Tuple
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def load_digits_splits(seed: int = 42) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    X, y = load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8).astype(np.float32) / 16.0
    mean = X.mean()
    std = X.std() + 1e-6
    X = (X - mean) / std
    y = y.astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    def _dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(features[:, None], dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    return _dataset(X_train, y_train), _dataset(X_val, y_val), _dataset(X_test, y_test)
