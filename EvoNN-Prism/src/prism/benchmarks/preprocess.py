"""Simple numpy-based preprocessing (no sklearn dependency)."""

from __future__ import annotations

import numpy as np

_MIN_STD = 1e-8


class Preprocessor:
    """StandardScaler-style preprocessing using only numpy."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.fitted: bool = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Compute mean/std from X and return standardized copy."""
        X = self._as_preprocessed_float32(X)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < _MIN_STD] = 1.0
        self.fitted = True

        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply stored mean/std to X."""
        mean, std = self._fitted_parameters()
        X = self._as_preprocessed_float32(X)

        return (X - mean) / std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        mean, std = self._fitted_parameters()
        return X * std + mean

    def reset(self) -> None:
        """Clear fitted state."""
        self.mean = None
        self.std = None
        self.fitted = False

    def _fitted_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        return self.mean, self.std

    @staticmethod
    def _as_preprocessed_float32(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=True)
        Preprocessor._fill_nans_with_column_medians(X)
        return X

    @staticmethod
    def _fill_nans_with_column_medians(X: np.ndarray) -> None:
        nan_mask = np.isnan(X)
        if not np.any(nan_mask):
            return

        col_medians = np.nanmedian(X, axis=0)
        for col_idx in range(X.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            if np.any(col_nan_mask):
                X[col_nan_mask, col_idx] = col_medians[col_idx]
