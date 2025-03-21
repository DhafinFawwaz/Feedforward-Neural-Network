"""
Cara penggunaan:
init = WeightInitiator(init_method="zero", nodes=np.array([2, 1, 3]), lower_bound=4, upper_bound=7)
init = WeightInitiator(init_method="uniform", nodes=np.array([2, 1, 3]), lower_bound=4, upper_bound=7)
init = WeightInitiator(init_method="normal", nodes=np.array([2, 1, 3]), mean=24, std=2, seed=69)
weights = init.get_weights()
biases = init.get_bias()
grad_weight = initiator.get_gradient_weights()
grad_bias = initiator.get_gradient_bias()
"""

import random
import numpy as np
from numpy.typing import NDArray
from typing import Literal

class WeightInitiator:
    def __init__(
            self, 
            init_method:Literal["zero","uniform","normal"],
            nodes:NDArray,
            # layers: int, 
            lower_bound: float = 0.0, 
            upper_bound: float = 1.0, 
            mean: float = 0.0, 
            std: float = 1.0, 
            seed:int = None
        ) -> None:
        if nodes is None:
            raise ValueError("Nodes must be specified")
        
        self.layered_weights = []
        self.bias = []
        self.nodes = nodes
        self.init_method = init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed


    def _zero_init(self, neurons_before: int, neurons: int):
        return np.matrix([[0.0 for _ in range(neurons)] for _ in range(neurons_before)])

    def _zero_init_bias(self, neurons: int, bias: bool):
        if bias:
            return np.array([0.0 for _ in range(neurons)])
    
    def _uniform_init(self, neurons_before: int, neurons: int, lower_bound: float, upper_bound: float):
        if lower_bound is None or upper_bound is None:
            raise ValueError("Lower Bound and Upper Bound must be specified")
        if lower_bound > upper_bound:
            raise ValueError("Lower Bound must be less than Upper Bound")
        return np.matrix([[random.uniform(lower_bound, upper_bound) for _ in range(neurons)] for _ in range(neurons_before)])

    def _uniform_init_bias(self, lower_bound: float, upper_bound: float, neurons: int):
        return np.array([random.uniform(lower_bound, upper_bound) for _ in range(neurons)])
    
    def _normal_init(self, neurons_before: int, neurons: int, mean: float, std: float, seed:int = None):
        if mean is None or std is None:
            raise ValueError("Mean and Standard Deviation must be specified")
        if seed is not None:
            random.seed(seed)
            return np.matrix([[random.gauss(mean, std) for _ in range(neurons)] for _ in range(neurons_before)])
        else:
            return np.matrix([[random.gauss(mean, std) for _ in range(neurons)] for _ in range(neurons_before)])

    def _normal_init_bias(self, neurons: int, mean: float, std: float, seed:int = None):
        if mean is None or std is None:
            raise ValueError("Mean and Standard Deviation must be specified")
        if seed is not None:
            random.seed(seed)
            return np.array([random.gauss(mean, std) for _ in range(neurons)])
        else:
            return np.array([random.gauss(mean, std) for _ in range(neurons)])

    def get_weights(self):
        for i in range(len(self.nodes)-1):
            if self.init_method == "zero":
                weights = self._zero_init(neurons_before=int(self.nodes[i]), neurons= int(self.nodes[i+1]))
                self.layered_weights.append(weights)
            elif self.init_method == "uniform":
                weights = self._uniform_init(neurons_before=int(self.nodes[i]), neurons= int(self.nodes[i+1]), lower_bound=self.lower_bound, upper_bound=self.upper_bound)
                self.layered_weights.append(weights)
            elif self.init_method == "normal":
                weights = self._normal_init(neurons_before=int(self.nodes[i]), neurons= int(self.nodes[i+1]), mean=self.mean, std=self.std, seed=self.seed)
                self.layered_weights.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.layered_weights
    
    def get_bias(self):
        for i in range(1,len(self.nodes)):
            if self.init_method == "zero":
                weights = self._zero_init_bias(neurons=int(self.nodes[i]), bias=True)
                self.bias.append(weights)
            elif self.init_method == "uniform":
                weights = self._uniform_init_bias(neurons=int(self.nodes[i]), lower_bound=self.lower_bound, upper_bound=self.upper_bound)
                self.bias.append(weights)
            elif self.init_method == "normal":
                weights = self._normal_init_bias(neurons=int(self.nodes[i]), mean=self.mean, std=self.std, seed=self.seed)
                self.bias.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.bias
    
    def get_gradient_weights(self):
        initiator = WeightInitiator(init_method="zero", nodes=self.nodes)
        return initiator.get_weights()
    
    def get_gradient_bias(self):
        initiator = WeightInitiator(init_method="zero", nodes=self.nodes)
        return initiator.get_bias()

if __name__ == "__main__":
    init = WeightInitiator(init_method="zero", nodes=np.array([2, 1, 3]), lower_bound=4, upper_bound=7)
    # init = WeightInitiator(init_method="uniform", nodes=np.array([2, 1, 3]), lower_bound=4, upper_bound=7)
    # init = WeightInitiator(init_method="normal", nodes=np.array([2, 1, 3]), mean=24, std=2, seed=69)
    weights = init.get_weights()
    biases = init.get_bias()

    for weight in weights:
        for row in weight:
            print(row)
        print("="*50)

    print("="*30 + "BIAS" + "="*30)
    for bias in biases:
        print(bias)
        print("=" * 50)
