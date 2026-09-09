import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        output = np.dot(X,weights)
        return np.round(output,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        squared_error = 0
        for i in range(len(model_prediction)):
            squared_error += ((model_prediction[i] - ground_truth[i])**2) 
        MSE = squared_error / len(model_prediction)
        # Round to 5 decimal places
        return np.round(MSE,5)[0]
