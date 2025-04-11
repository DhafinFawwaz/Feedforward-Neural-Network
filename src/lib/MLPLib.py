from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
from sklearn.utils._indexing import _safe_indexing
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer

_STOCHASTIC_SOLVERS = ["sgd", "adam"]

import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import Literal
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import ACTIVATIONS, LOSS_FUNCTIONS, DERIVATIVES

class MLPLIB(MLPClassifier):
    def __init__(
        self,
        init_method: Literal["zero", "uniform", "normal"] = "zero",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = None,
        verbose: bool = True,
        alpha: float = 0,
        alpha_l1: float = 0,
        dtype: type = "float32",
        **kwargs
    ):
        
        super().__init__(
            **kwargs, 
            shuffle=False,
            alpha=alpha,
            solver='sgd',
            momentum=0.0,
            nesterovs_momentum=False,
            early_stopping=False,
            learning_rate='constant',
            random_state=seed,
            verbose=verbose
        )
        self.init_method = init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed
        self.alpha_l1 = alpha_l1
        self.dtype = dtype
        # self.alpha_l2 is self.alpha

    # Override to set the seed
    def _initialize(self, y, layer_units, dtype):
        if self.seed is not None:
            np.random.seed(self.seed)
        super()._initialize(y, layer_units, dtype)
        
    
    # Override to include custom weight initialization
    def _init_coef(self, fan_in, fan_out, dtype):
        """Custom weight initialization based on `init_method`."""
        if self.activation == 'logistic':
            init_bound = np.sqrt(2.0 / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

        if self.init_method == 'zero':
            coef_init = np.zeros((fan_in, fan_out), dtype=dtype)
            intercept_init = np.zeros(fan_out, dtype=dtype)
        elif self.init_method == 'uniform':
            coef_init = np.random.uniform(self.lower_bound, self.upper_bound, (fan_in, fan_out)).astype(dtype)
            intercept_init = np.random.uniform(self.lower_bound, self.upper_bound, fan_out).astype(dtype)
        elif self.init_method == 'normal':
            intercept_init = np.random.normal(self.mean, self.std, fan_out).astype(dtype)
            coef_init = np.random.normal(self.mean, self.std, (fan_in, fan_out)).astype(dtype)
        elif self.init_method == 'xavier_uniform':
            bound_limit = np.sqrt(6 / (fan_in + fan_out))
            coef_init = np.random.uniform(-bound_limit, bound_limit, (fan_in, fan_out)).astype(self.dtype)
            intercept_init = np.random.uniform(-bound_limit, bound_limit, fan_out).astype(self.dtype)
        elif self.init_method == 'he_uniform':
            bound_limit = np.sqrt(6 / fan_in)
            coef_init = np.random.uniform(-bound_limit, bound_limit, (fan_in, fan_out)).astype(self.dtype)
            intercept_init = np.random.uniform(-bound_limit, bound_limit, fan_out).astype(self.dtype)
        elif self.init_method == 'xavier_normal':
            deviation = np.sqrt(2 / (fan_in + fan_out))
            intercept_init = np.random.normal(0, deviation, fan_out).astype(self.dtype)
            coef_init = np.random.normal(0, deviation, (fan_in, fan_out)).astype(self.dtype)
        elif self.init_method == 'he_normal':
            deviation = np.sqrt(2 / fan_in)
            intercept_init = np.random.normal(0, deviation, fan_out).astype(self.dtype)
            coef_init = np.random.normal(0, deviation, (fan_in, fan_out)).astype(self.dtype)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        # print("MLPLib coef_init")
        # print(coef_init)
        # print("MLPLib intercept_init")
        # print(intercept_init)
        return coef_init, intercept_init
    

    # Override to include L1 regularization
    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        loss, coef_grads, intercept_grads = super()._backprop(X, y, activations, deltas, coef_grads, intercept_grads)

        # L1 regularization
        n_samples = X.shape[0]
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.sum(np.abs(s))
        loss += (0.5 * self.alpha_l1) * values / n_samples

        return loss, coef_grads, intercept_grads
    
    # Override to include L1 regularization
    def _compute_loss_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer] # L2 regularization
        coef_grads[layer] += self.alpha_l1 * np.sign(self.coefs_[layer]) # L1 regularization
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)