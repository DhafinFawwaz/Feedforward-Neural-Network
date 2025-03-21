"""
Cara penggunaan:
initiator = WeightInitiatior(init_method=WeightInitMethod.NORMAL, nodes=[2,3,1], mean=24, std=2, seed=69)
initiator = WeightInitiatior(init_method=WeightInitMethod.UNIFORM, nodes=[2,3,1], lower_bound=4, upper_bound=7)
initiator = WeightInitiatior(init_method=WeightInitMethod.ZERO, nodes=[2,3,1])
weights = initiator.get_weights()
bias = initiator.get_bias()
grad_weight = initiator.get_gradient_weights()
grad_bias = initiator.get_gradient_bias()
"""

import random
import enum
from typing import Literal

class WeightInitiator:
    def __init__(
            self, 
            init_method:Literal["zero","uniform","normal"],
            nodes:list[int], 
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


    def _zero_init(self, neurons_before: int, neurons: int) -> list[list[float]]:
        return [[0.0 for _ in range(neurons_before)] for _ in range(neurons)]

    def _zero_init_bias(self, neurons: int, bias: bool):
        if bias:
            return [0.0 for _ in range(neurons)]
    
    def _uniform_init(self, neurons_before: int, neurons: int, lower_bound: float, upper_bound: float) -> list[list[float]]:
        if lower_bound is None or upper_bound is None:
            raise ValueError("Lower Bound and Upper Bound must be specified")
        if lower_bound > upper_bound:
            raise ValueError("Lower Bound must be less than Upper Bound")
        return [[random.uniform(lower_bound, upper_bound) for _ in range(neurons_before)] for _ in range(neurons)]

    def _uniform_init_bias(self, lower_bound: float, upper_bound: float, neurons: int):
        return [random.uniform(lower_bound, upper_bound) for _ in range(neurons)]
    
    def _normal_init(self, neurons_before: int, neurons: int, mean: float, std: float, seed:int = None) -> list[list[float]]:
        if mean is None or std is None:
            raise ValueError("Mean and Standard Deviation must be specified")
        if seed is not None:
            random.seed(seed)
            return [[random.gauss(mean, std) for _ in range(neurons_before)] for _ in range(neurons)]
        else:
            return [[random.gauss(mean, std) for _ in range(neurons_before)] for _ in range(neurons)]

    def _normal_init_bias(self, neurons: int, mean: float, std: float, seed:int = None)-> list[float]:
        if mean is None or std is None:
            raise ValueError("Mean and Standard Deviation must be specified")
        if seed is not None:
            random.seed(seed)
            return [random.gauss(mean, std) for _ in range(neurons)]
        else:
            return [random.gauss(mean, std) for _ in range(neurons)]

    def get_weights(self):
        for i in range(len(self.nodes)-1):
            if self.init_method == WeightInitMethod.ZERO:
                weights = self._zero_init(neurons_before=self.nodes[i], neurons= self.nodes[i+1])
                self.layered_weights.append(weights)
            elif self.init_method == WeightInitMethod.UNIFORM:
                weights = self._uniform_init(neurons_before=self.nodes[i], neurons=self.nodes[i+1], lower_bound=self.lower_bound, upper_bound=self.upper_bound)
                self.layered_weights.append(weights)
            elif self.init_method == WeightInitMethod.NORMAL:
                weights = self._normal_init(neurons_before=self.nodes[i], neurons=self.nodes[i+1], mean=self.mean, std=self.std, seed=self.seed)
                self.layered_weights.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.layered_weights
    
    def get_bias(self):
        for i in range(1,len(self.nodes)):
            if self.init_method == WeightInitMethod.ZERO:
                weights = self._zero_init_bias(neurons=self.nodes[i+1], bias=True)
                self.bias.append(weights)
            elif self.init_method == WeightInitMethod.UNIFORM:
                weights = self._uniform_init_bias(neurons=self.nodes[i], lower_bound=self.lower_bound, upper_bound=self.upper_bound)
                self.bias.append(weights)
            elif self.init_method == WeightInitMethod.NORMAL:
                weights = self._normal_init_bias(neurons=self.nodes[i], mean=self.mean, std=self.std, seed=self.seed)
                self.bias.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.bias
    
    def get_gradient_weights(self):
        initiator = WeightInitiator(init_method=WeightInitMethod.ZERO, nodes=self.nodes)
        return initiator.get_weights()
    
    def get_gradient_bias(self):
        initiator = WeightInitiator(init_method=WeightInitMethod.ZERO, nodes=self.nodes)
        return initiator.get_bias()