import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        clipped_pred = np.clip(y_pred, a_min = 10**(-7), a_max = 1 - 10**(-7))

        bce = - (1 / len(y_true)) * np.sum(y_true * np.log(clipped_pred) + (1 - y_true) * np.log(1 - clipped_pred))
        return round(float(bce), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        clipped_pred = np.clip(y_pred, a_min = 10**(-7), a_max = 1 - 10**(-7))
        cce = -(1/(len(y_true)) * np.sum(y_true * np.log(clipped_pred))) 
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        return round(float(cce), 4)
