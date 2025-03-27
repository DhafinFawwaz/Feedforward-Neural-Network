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
        
        # print("MLPLib coef_init")
        # print(coef_init)
        # print("MLPLib intercept_init")
        # print(intercept_init)
        return coef_init, intercept_init
    
    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        # print("MLPLib self.coefs_")
        # print(self.coefs_)
        # print("MLPLib self.intercepts_")
        # print(self.intercepts_)

        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            # print("dot")
            # print(activations[i])
            # print(self.coefs_[i])
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])


        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])

        # print(activations[-1][-1])
        return activations


    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        print(self.coefs_)
        activations = self._forward_pass(activations)
        # print(activations[-1])

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # print("y_act", y)
        # print("y_pred", activations[-1])
        # print("loss", loss)

        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )
            # print(i-1, coef_grads[i-1])

        # print(coef_grads)
        # print(intercept_grads)
        return loss, coef_grads, intercept_grads
    
    def _compute_loss_grad(
        self, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        # coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] = activations[layer].T @ deltas[layer]
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)


    


    def _fit_stochastic(
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
    ):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            # elif self.solver == "adam":
            #     self._optimizer = AdamOptimizer(
            #         params,
            #         self.learning_rate_init,
            #         self.beta_1,
            #         self.beta_2,
            #         self.epsilon,
            #     )

        # early_stopping in partial_fit doesn't make sense
        if self.early_stopping and incremental:
            raise ValueError("partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                random_state=self._random_state,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            # if self.batch_size > n_samples:
            #     warnings.warn(
            #         "Got `batch_size` less than 1 or larger than "
            #         "sample size. It is going to be clipped"
            #     )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                # if self.shuffle:
                #     # Only shuffle the sample indices instead of X and y to
                #     # reduce the memory footprint. These indices will be used
                #     # to slice the X and y.
                #     sample_idx = shuffle(sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        X_batch = _safe_indexing(X, sample_idx[batch_slice])
                        y_batch = y[sample_idx[batch_slice]]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch,
                        y_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )

                    # update weights
                    grads = coef_grads + intercept_grads
                    # print("before")
                    # print(self.coefs_)
                    self._optimizer.update_params(params, grads)
                    # print("after")
                    # print(self.coefs_)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                # if self.n_iter_ == self.max_iter:
                #     warnings.warn(
                #         "Stochastic Optimizer: Maximum iterations (%d) "
                #         "reached and the optimization hasn't converged yet."
                #         % self.max_iter,
                #         ConvergenceWarning,
                #     )
        except KeyboardInterrupt:
            pass
            # warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts