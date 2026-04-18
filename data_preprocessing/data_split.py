"""
data_preprocessing/data_split.py

Contains the function to split data into train/validation sets
using scikit-learn's train_test_split or GroupShuffleSplit.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_data(X, Y, test_size=0.2, random_state=42, groups=None):
    """
    Split the dataset into train and validation sets.
    :param X: Input features, shape (N, input_dim)
    :param Y: Output targets, shape (N, output_dim)
    :param test_size: Fraction of data for validation
    :param random_state: For reproducibility
    :param groups: Optional group labels. If provided, samples from the same
        group are kept in the same split.
    :return: X_train, X_val, Y_train, Y_val
    """
    if groups is None:
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)

    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(X):
        raise ValueError(
            f"Length mismatch: len(groups)={len(groups_arr)} but len(X)={len(X)}."
        )
    unique_groups = np.unique(groups_arr)
    if unique_groups.size < 2:
        print("[WARN] Not enough unique groups for grouped split; fallback to random split.")
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, Y, groups=groups_arr))
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]
