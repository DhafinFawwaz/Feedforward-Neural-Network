"""
Cara penggunaan:
clf = MLPLIB(
    hidden_layer_sizes=[10],
    max_iter=20,
    random_state=42,
    learning_rate_init=0.01,
    init_method="normal",
)

clf.fit(X_train_scaled, y_train)
sk_pred = clf.predict(X_test_scaled)
sk_accuracy = accuracy_score(y_test, sk_pred)
print("[SKLEARN] Prediction: ",sk_pred)
print("[SKLEARN] Accuracy: ", sk_accuracy)

-----------------------------
import random

weight_configs =  ['normal', 'zero', 'uniform']

print("Width Variations Experiment:")
for weight_config in weight_configs:
    print(f"\nTesting weight configuration: {weight_config}")

    lower_bound=5.39294405e-05
    upper_bound=1
    mean=5.39294405e-05
    std=.44
    seed=69

    # Scikit-learn MLP
    sk_mlp = MLPLIB(
        hidden_layer_sizes=[10],
        max_iter=20,
        random_state=42,
        learning_rate_init=0.01,
        init_method="normal",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mean=mean,
        std=std,
        seed=seed
    )



    # Custom MLP
    custom_mlp = FFNNClassifier(
        max_epoch=20,
        learning_rate=0.01,
        hidden_layer_sizes=[10],
        init_method="normal",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        mean=mean,
        std=std,
        seed=seed
    )
    custom_mlp.fit(X_train_scaled, y_train_one_hot)


    model_comparison(sk_mlp, custom_mlp, False)
"""
from sklearn.base import is_classifier

_STOCHASTIC_SOLVERS = ["sgd", "adam"]

import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import Literal
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import ACTIVATIONS

class MLPLIB(MLPClassifier):
    def __init__(
        self,
        init_method: Literal["zero", "uniform", "normal"] = "zero",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = None,
        **kwargs
    ):
        
        super().__init__(
            **kwargs, 
            shuffle=False,
            alpha=0.0,
            solver='sgd',
            momentum=0.0,
            nesterovs_momentum=False,
            early_stopping=False,
            learning_rate='constant',
            random_state=seed,
            verbose=True
        )
        self.init_method = init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _initialize(self, y, layer_units, dtype):
        # set all attributes, allocate weights etc. for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        if not is_classifier(self):
            self.out_activation_ = "identity"
        # Output for multi class
        elif self._label_binarizer.y_type_ == "multiclass":
            self.out_activation_ = "softmax"
        # Output for binary class and multi-label
        else:
            self.out_activation_ = "logistic"

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        if self.seed is not None:
            np.random.seed(self.seed)

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self._best_coefs = [c.copy() for c in self.coefs_]
        self._best_intercepts = [i.copy() for i in self.intercepts_]

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf
                self.validation_scores_ = None
                self.best_validation_score_ = None
        
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
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        
        print("MLPLib coef_init")
        print(coef_init)
        print("MLPLib intercept_init")
        print(intercept_init)
        return coef_init, intercept_init
    
    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """

        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            # print(i,"dot")
            # print(activations[i])
            # print(self.coefs_[i])
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            # print("dot",activations[i + 1])
            # print("self.intercepts_[i]", self.intercepts_[i])
            activations[i + 1] += self.intercepts_[i]
            # print("activations[i + 1]",activations[i + 1])

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])


        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])


        return activations
