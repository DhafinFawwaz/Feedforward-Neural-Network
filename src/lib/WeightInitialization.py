import numpy as np
import math
from typing import List, Union, Literal

class WeightInitialization:
    def __init__(
        self, 
        layer_units: List[int],
        init_method:Literal["uniform", "normal","zero"],
        lower_bound=5.39294405e-05,
        upper_bound = 1,
        mean = 5.39294405e-05,
        std = .44,
        seed = 69,
        dtype: type = "float32", # becase MLPClassifier uses float32
    ):
        self.layer_units = layer_units
        self.init_method = init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed
        self.dtype = dtype

        self.coefs_ = []
        self.intercepts_ = []
    
    def _init_coef(self, fan_in: int, fan_out: int) -> tuple:
        """
        Initialize coefficients and intercepts for a layer
        
        Parameters:
        -----------
        n_fan_in : int
            Number of input neurons
        n_fan_out : int
            Number of output neurons
        
        Returns:
        --------
        tuple: (coefficient matrix, intercept vector)
        """
        # Determine initialization scale based on activation and fan-in/fan-out
        if self.init_method == 'zero':
            coef_init = np.zeros((fan_in, fan_out), dtype=self.dtype)
            intercept_init = np.zeros(fan_out, dtype=self.dtype)
        elif self.init_method == 'uniform':
            coef_init = np.random.uniform(self.lower_bound, self.upper_bound, (fan_in, fan_out)).astype(self.dtype)
            intercept_init = np.random.uniform(self.lower_bound, self.upper_bound, fan_out).astype(self.dtype)
        elif self.init_method == 'normal':
            intercept_init = np.random.normal(self.mean, self.std, fan_out).astype(self.dtype)
            coef_init = np.random.normal(self.mean, self.std, (fan_in, fan_out)).astype(self.dtype)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # print("weightinit coef_init")
        # print(coef_init)
        # print("weightinit intercept_init")
        # print(intercept_init)
        return coef_init, intercept_init
        
    
        return coef, intercept
    
    def initialize_weights(self):
        """
        Generate weights for all layers
        
        Returns:
        --------
        tuple: (list of coefficient matrices, list of intercept vectors)
        """
        self.coefs_ = []
        self.intercepts_ = []

        if self.seed is not None:
            np.random.seed(self.seed)

        for i in range(len(self.layer_units) - 1):
            coef, intercept = self._init_coef(
                self.layer_units[i], 
                self.layer_units[i + 1]
            )
            self.coefs_.append(coef)
            self.intercepts_.append(intercept)
        
        return self.coefs_, self.intercepts_

# Example usage
if __name__ == "__main__":
    # Example: Initialize weights for a network with layers [2, 10, 5, 3]
    initiator = WeightInitialization(
        layer_units=[2, 10, 5, 3],
        activation='relu'
    )
    
    coefs, intercepts = initiator.initialize_weights()
    
    # Print details of initialized weights
    for i, (coef, intercept) in enumerate(zip(coefs, intercepts), 1):
        print(f"Layer {i} Coefficient Shape: {coef.shape}")
        print(f"Layer {i} Intercept Shape: {intercept.shape}")
        print("-" * 40)