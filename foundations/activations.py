import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:

        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        sigmoid_output = 1 / (1 + np.exp(-z))
        return np.round(sigmoid_output, 5)
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu_output = np.maximum(0,z)
        return np.round(relu_output, 5)
