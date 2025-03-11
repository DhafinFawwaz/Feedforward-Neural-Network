"""
Cara penggunaan:
initiator = WeightInitiatior(init_method=WeightInitMethod.NORMAL, neurons=2, layers=3, mean=24, std=2, seed=69)
initiator = WeightInitiatior(WeightInitMethod.UNIFORM, neurons=2, layers=3, lower_bound=4, upper_bound=7)
initiator = WeightInitiatior(WeightInitMethod.ZERO, neurons=2, layers=3)
weights,bias = initiator.get_weights()
"""

# TODO: Bobot Gradien
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
            neurons:int, 
            layers: int, 
            lower_bound: float = 0.0, 
            upper_bound: float = 1.0, 
            mean: float = 0.0, 
            std: float = 1.0, 
            seed:int = None
        ) -> None:
        self.layered_weights = []
        self.bias = []
        for i in range(layers+1):
            weights = []
            if init_method == WeightInitMethod.ZERO:
                weights = self.zero_init(neurons)
            elif init_method == WeightInitMethod.UNIFORM:
                weights = self.uniform_init(neurons, lower_bound, upper_bound)
            elif init_method == WeightInitMethod.NORMAL:
                weights = self.normal_init(neurons, mean, std, seed)

            if i == layers:
                self.bias = weights
            else:
                self.layered_weights.append(weights)

    def zero_init(self, neurons: int) -> list[list[float]]:
        return [0.0] * neurons
    
    def uniform_init(self, neurons: int, lower_bound: float, upper_bound: float) -> list[list[float]]:
        return [random.uniform(lower_bound, upper_bound) for _ in range(neurons)]
    
    def normal_init(self, neurons: int, mean: float, std: float, seed:int = None) -> list[list[float]]:
        if seed is not None:
            random.seed(seed)
            return [random.gauss(mean, std) for _ in range(neurons)]
        else:
            return [random.gauss(mean, std) for _ in range(neurons)]
        
    def get_weights(self) -> tuple[list[list[list[float]]],list[float]]:
        return self.layered_weights, self.bias