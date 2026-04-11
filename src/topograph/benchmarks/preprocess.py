"""Simple numpy-based preprocessing (no sklearn dependency)."""

from __future__ import annotations

import numpy as np


class Preprocessor:
    """StandardScaler-style preprocessing using only numpy."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.fitted: bool = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Compute mean/std from X and return standardized copy."""
        X = X.astype(np.float32, copy=True)

        # Handle NaN values before computing stats
        if np.any(np.isnan(X)):
            col_medians = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                if np.any(mask):
                    X[mask, j] = col_medians[j]

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < 1e-8] = 1.0
        self.fitted = True

        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply stored mean/std to X."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        X = X.astype(np.float32, copy=True)

        if np.any(np.isnan(X)):
            col_medians = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                if np.any(mask):
                    X[mask, j] = col_medians[j]

        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        return X * self.std + self.mean

    def reset(self) -> None:
        """Clear fitted state."""
        self.mean = None
        self.std = None
        self.fitted = False
