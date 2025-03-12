"""
Cara penggunaan:
initiator = WeightInitiatior(init_method=WeightInitMethod.NORMAL, neurons=2, layers=3, mean=24, std=2, seed=69)
initiator = WeightInitiatior(init_method=WeightInitMethod.UNIFORM, neurons=2, layers=3, lower_bound=4, upper_bound=7)
initiator = WeightInitiatior(init_method=WeightInitMethod.ZERO, neurons=2, layers=3)
weights = initiator.get_weights()
bias = initiator.get_bias()
grad_weight = initiator.get_gradient_weights()
grad_bias = initiator.get_gradient_bias()
"""

import random
import enum

class WeightInitMethod(enum.Enum):
    ZERO = "zero"
    UNIFORM = "uniform"
    NORMAL = "normal"

class WeightInitiatior:
    def __init__(
            self, 
            init_method:WeightInitMethod, 
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
        

    def zero_init(self, neurons_before: int, neurons: int, bias: bool) -> list[list[float]]:
        if (bias):
            return [0.0 for _ in range(neurons)]
        return [[0.0 for _ in range(neurons_before)] for _ in range(neurons)]
    
    def uniform_init(self, neurons_before: int, neurons: int, lower_bound: float, upper_bound: float, bias: bool) -> list[list[float]]:
        if lower_bound == None or upper_bound == None:
            raise ValueError("Lower Bound and Upper Bound must be specified")
        if lower_bound > upper_bound:
            raise ValueError("Lower Bound must be less than Upper Bound")
        if bias:
            return [random.uniform(lower_bound, upper_bound) for _ in range(neurons)]
        return [[random.uniform(lower_bound, upper_bound) for _ in range(neurons_before)] for _ in range(neurons)]
    
    def normal_init(self, neurons_before: int, neurons: int, mean: float, std: float, bias: bool, seed:int = None) -> list[list[float]]:
        if mean == None or std == None:
            raise ValueError("Mean and Standard Deviation must be specified")
        if seed is not None:
            random.seed(seed)
            if bias:
                return [random.gauss(mean, std) for _ in range(neurons)]
            else:
                return [[random.gauss(mean, std) for _ in range(neurons_before)] for _ in range(neurons)]
        else:
            if bias:
                return [random.gauss(mean, std) for _ in range(neurons)]
            else:
                return [[random.gauss(mean, std) for _ in range(neurons_before)] for _ in range(neurons)]
        
    def get_weights(self):
        for i in range(len(self.nodes)-1):
            if self.init_method == WeightInitMethod.ZERO:
                weights = self.zero_init(neurons_before=self.nodes[i], neurons= self.nodes[i+1], bias=False)
                self.layered_weights.append(weights)
            elif self.init_method == WeightInitMethod.UNIFORM:
                weights = self.uniform_init(neurons_before=self.nodes[i], neurons=self.nodes[i+1], lower_bound=self.lower_bound, upper_bound=self.upper_bound, bias=False)
                self.layered_weights.append(weights)
            elif self.init_method == WeightInitMethod.NORMAL:
                weights = self.normal_init(neurons_before=self.nodes[i], neurons=self.nodes[i+1], mean=self.mean, std=self.std, seed=self.seed, bias=False)
                self.layered_weights.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.layered_weights
    
    def get_bias(self):
        for i in range(1,len(self.nodes)):
            if self.init_method == WeightInitMethod.ZERO:
                weights = self.zero_init(neurons_before=self.nodes[i], neurons= self.nodes[i], bias=True)
                self.bias.append(weights)
            elif self.init_method == WeightInitMethod.UNIFORM:
                weights = self.uniform_init(neurons_before=self.nodes[i], neurons=self.nodes[i], lower_bound=self.lower_bound, upper_bound=self.upper_bound, bias=True)
                self.bias.append(weights)
            elif self.init_method == WeightInitMethod.NORMAL:
                weights = self.normal_init(neurons_before=self.nodes[i], neurons=self.nodes[i], mean=self.mean, std=self.std, seed=self.seed, bias=True)
                self.bias.append(weights)
            else:
                raise ValueError("Invalid Weight Initialization Method")
        return self.bias
    
    def get_gradient_weights(self):
        initiator = WeightInitiatior(init_method=WeightInitMethod.ZERO, nodes=self.nodes)
        return initiator.get_weights()
    
    def get_gradient_bias(self):
        initiator = WeightInitiatior(init_method=WeightInitMethod.ZERO, nodes=self.nodes)
        return initiator.get_bias()