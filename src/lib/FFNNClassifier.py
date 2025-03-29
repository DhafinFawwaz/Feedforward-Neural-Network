from typing import Literal, Callable, Union, List
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pickle

from sklearn.calibration import expit
from scipy.special import xlogy
from lib.WeightInitialization import WeightInitialization
from scipy.special import expit


class FFNNClassifier:
    def __init__(self,
            hidden_layer_sizes: NDArray,
            learning_rate: float,
            activation_func: List[Literal['linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'softsign', 'softplus']] = None,
            verbose: int = 0, # 0: no print, 1: print epoch progress
            max_epoch: int = 50,
            batch_size: int = 256,
            loss_func: Literal['mean_squared_error', 'binary_cross_entropy', 'categorical_cross_entropy'] = 'categorical_cross_entropy',
            init_method: Literal['normal', 'zero', 'uniform',"xavier_normal","xavier_uniform","he_normal","he_uniform"] = 'zero',
            lower_bound: float = 0.0,
            upper_bound: float = 1.0,
            mean: float = 0.0,
            std: float = 1.0,
            seed: int | None = None,

            l1: float = 0.0,
            l2: float = 0.0,
        ):
        self.l1 = l1
        self.l2 = l2
        self.hidden_layer_sizes = hidden_layer_sizes
        self.X: NDArray = []
        self.y: list[ArrayLike] = []
        self.weights_history: list[NDArray] = [] # array of weight matrix. index is current epoch
        self.biases_history: list[ArrayLike] = [] # array of bias list. index is current epoch
        self.weight_gradients_history: list[NDArray] = [] # array of weight gradients. index is current epoch

        if activation_func is None:
            self.activation_func = ['sigmoid'] * len(hidden_layer_sizes) + ['softmax']
        else:
            self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epoch_amount = max_epoch
        self.batch_size = batch_size
        self.loss_func = loss_func

        self.init_method = init_method
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std
        self.seed = seed

        self.amount_of_features = -1
        self.amount_of_classes = -1

        self.loss_history = []
        self.validation_loss_history = []



        if len(self.hidden_layer_sizes) != len(self.activation_func) - 1:
            raise Exception("should be len(hidden_layer_sizes) == len(activation_func) - 1")
        
        if self.loss_func != 'mean_squared_error' and self.loss_func != 'binary_cross_entropy' and self.loss_func != 'categorical_cross_entropy':
            raise Exception("loss_func should be either 'mean_squared_error', 'binary_cross_entropy', or 'categorical_cross_entropy'")
        
        if self.init_method not in ['normal', 'zero', 'uniform',"xavier_normal","xavier_uniform","he_normal","he_uniform"]:
            raise Exception("init_method should be either 'normal', 'zero', 'uniform', 'xavier_[normal/uniform]', or 'he_[normal/uniform]'")
        
        for i in range(len(self.activation_func)):
            if self.activation_func[i] != 'linear' and self.activation_func[i] != 'relu' and self.activation_func[i] != 'sigmoid' and self.activation_func[i] != 'tanh' and self.activation_func[i] != 'softmax' and self.activation_func[i] != 'softsign' and self.activation_func[i] != 'softplus':
                raise Exception("activation_func should be either 'linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'softsign', or 'softplus'")

    # return [ matrix, matrix, matrix ... ] where matrix is the weight adjacency matrix for each layer. length should be number of layers - 1 because its like the edges/connection between the nodes
    def _generate_initator_weights(self):
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        len_features = self._get_amount_of_features()
        layers = np.copy([len_features])
        len_classes = np.array([self._get_amount_of_classes()], dtype="int32")
        layers = np.append(layers, self.hidden_layer_sizes)
        layers = np.append(layers, len_classes)
        # print("layers:", layers)
        weight_initiator = WeightInitialization(
            init_method=self.init_method,
            layer_units=layers,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            mean=self.mean,
            std=self.std,
            seed=self.seed
        )
        coefs, intercepts = weight_initiator.initialize_weights()
        return coefs, intercepts

# region functions
    @staticmethod
    def _activation_function(x: Union[float, NDArray], func: str):
        if func == 'linear': return x
        elif func == 'relu': return np.maximum(0, x)
        # elif func == 'sigmoid': return 1.0/(1.0 + np.exp(-x))
        elif func == 'sigmoid': return expit(x) # to fix overflow issue
        elif func == 'tanh': return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True) # keepdims=True will keep the dimension of the original array
        elif func == "softsign":
            return x / (1 + np.abs(x))
        elif func == "softplus":
            return np.log(1 + np.exp(x))
        raise "Activation function not supported!"

    @staticmethod
    def _activation_derived_function(x: Union[float, NDArray], func: str):
        if func == 'linear': return np.ones_like(x, dtype="float32")
        elif func == 'relu': return np.where(x > 0, 1.0, 0.0).astype("float32")
        elif func == 'sigmoid':
            sig = FFNNClassifier._activation_function(x, 'sigmoid')
            return sig * (1 - sig)
        elif func == 'tanh':
            return 1 - np.tanh(x)**2
        elif func == 'softmax':
            batch_size, n = x.shape
            jacobians = np.zeros((batch_size, n, n), dtype="float32")

            for i in range(batch_size):
                s = x[i].reshape(-1, 1)
                jacobians[i] = np.diagflat(s) - s @ s.T

            return jacobians
        elif func == "softsign":
            one_plus_abs_x = 1 + np.abs(x)
            return 1 / (one_plus_abs_x * one_plus_abs_x)
        elif func == "softplus":
            # return 1 / (1 + np.exp(-x)) # sigmoid(x)
            return expit(x) # to fix overflow issue
        raise "Activation function not supported!"


    @staticmethod
    def _loss_function_derived(y_act, y_pred, func: str):
        if func == 'mean_squared_error': 
            return 2/len(y_act[0]) * (y_pred - y_act) # 2/len(y_act[0]) can actually be ignored because we have learning rate
        
        elif func == 'binary_cross_entropy': 
            return (y_pred - y_act)/(y_pred*(1 - y_pred)) * len(y_act[0]) # len(y_act[0]) can actually be ignored because we have learning rate
        
        elif func == 'categorical_cross_entropy': 
            return (y_pred - y_act)/len(y_act[0]) # len(y_act[0]) can actually be ignored because we have learning rate
        
    @staticmethod
    def _loss_function(y_act, y_pred, func: str):
        if func == "categorical_cross_entropy":
            eps = np.finfo(y_pred.dtype).eps
            y_pred = np.clip(y_pred, eps, 1 - eps)
            if y_pred.shape[1] == 1: y_pred = np.append(1 - y_pred, y_pred, axis=1) # case when the target has only 1 column. It should never happens tho but just put it here just in case.
            if y_act.shape[1] == 1: y_act = np.append(1 - y_act, y_act, axis=1)
            # return -(y_act*np.log(y_pred)).sum() / len(y_pred)

            # print("_loss_function")
            # print(y_pred)
            # print(-xlogy(y_act, y_pred))
            # print(-xlogy(y_act, y_pred).sum())
            # print(y_pred.shape[0])
            # print(-xlogy(y_act, y_pred).sum() / y_pred.shape[0])

            return -xlogy(y_act, y_pred).sum() / y_pred.shape[0]

        elif func == "mean_squared_error":
            res = (y_act - y_pred)
            return (res * res).mean()/2
        elif func == "binary_cross_entropy":
            return -((y_act*np.log(y_pred)).sum() + ((1 - y_act)*np.log(1 - y_pred)).sum())/ y_pred.shape[0]

        


# endregion functions


# region getters setters

    def _get_amount_of_features(self) -> int:
        if self.amount_of_features == -1: raise Exception("X and y is not set yet")
        return self.amount_of_features

    # Can only be called after setting X and y
    def _get_hidden_layer_sizes(self) -> np.typing.NDArray:

        len_features = self._get_amount_of_features()
        len_classes = self._get_amount_of_classes()
        layer_sizes = np.zeros(len(self.hidden_layer_sizes)+2, dtype="int32")
        layer_sizes[0] = len_features
        for i in range(1, len(self.hidden_layer_sizes)+1):
            layer_sizes[i] = self.hidden_layer_sizes[i-1]
        layer_sizes[len(layer_sizes)-1] = len_classes

        return layer_sizes

    # Can only be called after setting X and y
    def _generate_new_empty_layers(self):

        layer_sizes = self._get_hidden_layer_sizes()
        network_depth = len(layer_sizes)
        weights = []
        biases = []
        nodes = []
        nodes_active = []
        for i in range(network_depth-1):
            weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1]), dtype="float32"))
            biases.append(np.zeros(layer_sizes[i+1], dtype="float32"))
        for i in range(network_depth):
            nodes.append(np.zeros(layer_sizes[i], dtype="float32"))
            nodes_active.append(np.zeros(layer_sizes[i], dtype="float32"))

        return weights, nodes, nodes_active, biases


    # Can only be called after setting X and y
    def _get_amount_of_classes(self):
        # return len(np.unique(self.y))
        # return 10 # hardcoded because it messes up things when the possible value is not much
        # return len(self.y[0])
        return self.amount_of_classes

    def copy_list_as_zeros(self, list: list[NDArray]):
        return [np.zeros_like(w) for w in list]

# endregion getters setters



    def fit(self, X: NDArray, y: NDArray, X_test: NDArray, y_test: NDArray):
        if type(y) == list and type(y[0]) != list:
            y = np.array([[i] for i in y], dtype="int32")
        if type(y) == list and type(y[0]) == list:
            y = np.array(y, dtype="int32")

        if len(X) != len(y):
            raise Exception("length of X and y is not the same")
        if len(X) == 0:
            raise Exception("len(self.X) == 0")
        if len(X[0]) == 0:
            raise Exception("len(self.X[0]) == 0")
        if len(y) == 0:
            raise Exception("len(self.y) == 0")
        
        self.amount_of_features = len(X[0])
        self.amount_of_classes = len(y[0])
        self.X: NDArray = []
        self.y: ArrayLike = []
        self.weights_history: list[NDArray] = []
        self.biases_history: list[ArrayLike] = []
        self.weight_gradients_history: list[NDArray] = []
        self.loss_history = []

        self.X = X.astype("float32")
        self.y = y.astype("int32")
        initial_weight, initial_bias = self._generate_initator_weights()
        initial_gradients = [np.zeros_like(w, dtype="float32") for w in initial_weight]
        self.weights_history = initial_weight
        self.biases_history = initial_bias
        self.weight_gradients_history = initial_gradients
        layer_sizes = self._get_hidden_layer_sizes()
        network_depth = len(layer_sizes)

        for epoch in range(self.epoch_amount):
            total_loss = 0
            current_dataset_idx = 0
            while current_dataset_idx < len(self.X):
                weights, nodes, nodes_active, biases = self._generate_new_empty_layers() # will be filled with weights based on the previous epoch. Will be appended to the history after the end of the epoch

                # Feed Forward
                until_idx = min(current_dataset_idx+self.batch_size, len(self.X))
                nodes[0] = self.X[current_dataset_idx:until_idx]
                nodes_active[0] = self.X[current_dataset_idx:until_idx] # not passed to activation function for the first layer

                for k in range(1, network_depth):
                    w_k = self.weights_history[k-1]
                    b_k = self.biases_history[k-1]
                    h_k_min_1 = nodes_active[k-1]
                    a_k = b_k + (h_k_min_1 @ w_k) # numpy will automatically broadcast b_k (row will be copied to match the result from dot) so that this is addable

                    nodes[k] = a_k
                    nodes_active[k] = FFNNClassifier._activation_function(a_k, self.activation_func[k-1])
                
                y_act = self.y[current_dataset_idx:until_idx]
                y_pred = nodes_active[network_depth-1]
                
                loss = FFNNClassifier._loss_function( # cause the spec says so
                    y_act=y_act,
                    y_pred=y_pred,
                    func=self.loss_func
                )

                
                # L2 regularization
                total_l2 = 0
                for v in self.weights_history:
                    flat = v.ravel()
                    total_l2 += np.dot(flat, flat)
                loss += (0.5 * self.l2) * total_l2 / self.batch_size


                # L1 regularization
                total_l1 = 0
                for v in self.weights_history:
                    flat = v.ravel()
                    total_l1 += np.sum(np.abs(flat))
                loss += (0.5 * self.l1) * total_l1 / self.batch_size


                total_loss += loss * (until_idx - current_dataset_idx)


                # Backward Propagation
                weight_gradiens = [0 for i in range(len(self.weights_history))] # 0 will be replaced with numpy.array
                bias_gradiens = [0 for i in range(len(self.biases_history))] # 0 will be replaced with numpy.array


                delta = 0
                if (self.activation_func[-1] == 'softmax' and self.loss_func == 'categorical_cross_entropy') or (self.activation_func[-1] == "sigmoid" and self.loss_func == 'binary_cross_entropy') or (self.activation_func[-1] == "linear" and self.loss_func == 'mean_squared_error'):
                    delta = (nodes_active[-1] - self.y[current_dataset_idx:until_idx]).astype("float32")

                elif self.activation_func[-1] == 'softmax' and self.loss_func != 'categorical_cross_entropy':
                    jacobians = FFNNClassifier._activation_derived_function(nodes[-1], self.activation_func[-1])
                    loss_grad = FFNNClassifier._loss_function_derived(
                        y_act=y_act,
                        y_pred=y_pred,
                        func=self.loss_func
                    )
                    loss_grad_col = loss_grad[..., np.newaxis] # (batch_size, n) -> (batch_size, n, 1)
                    delta = np.matmul(jacobians, loss_grad_col)  # (batch_size, n, 1)
                    delta = np.squeeze(delta, axis=-1) # (batch_size, n)
                else:
                    # loss_grad = FFNNClassifier._loss_function_derived(
                    #     y_act=y_act,
                    #     y_pred=y_pred,
                    #     func=self.loss_func
                    # )
                    # delta = loss_grad * FFNNClassifier._activation_derived_function(nodes[-1], self.activation_func[-1])
                    delta = (nodes_active[-1] - self.y[current_dataset_idx:until_idx]).astype("float32")

                weight_gradiens[network_depth-2] = nodes_active[k-1].T @ delta
                weight_gradiens[network_depth-2] += self.l2 * self.weights_history[network_depth-2] # L2 regularization
                weight_gradiens[network_depth-2] += self.l1 * np.sign(self.weights_history[network_depth-2]) # L1 regularization
                weight_gradiens[network_depth-2] /= self.batch_size

                bias_gradiens[network_depth-2] = np.mean(delta, axis=0, keepdims=True)


                for k in range(network_depth-2, 0, -1): # from the last hidden layer (not including the output layer)
                    w = self.weights_history[k]

                    delta = np.dot(delta, w.T) * FFNNClassifier._activation_derived_function(nodes[k], self.activation_func[k-1])
                    weight_gradiens[k-1] = nodes_active[k-1].T @ delta
                    weight_gradiens[k-1] += self.l2 * self.weights_history[k-1] # L2 regularization
                    weight_gradiens[k-1] += self.l1 * np.sign(self.weights_history[k-1]) # L1 regularization
                    weight_gradiens[k-1] /= self.batch_size

                    bias_gradiens[k-1] = np.mean(delta, axis=0, keepdims=True)

                self.weight_gradients_history = weight_gradiens

                # Update
                for k in range(network_depth-1):
                    w_k = self.weights_history[k]
                    b_k = self.biases_history[k]

                    weights[k] = w_k - self.learning_rate * weight_gradiens[k]
                    biases[k] = b_k - self.learning_rate * bias_gradiens[k]

                self.weights_history = weights
                self.biases_history = biases

                current_dataset_idx += self.batch_size
            #### while loop ends here ##############################################

            prediction_with_validation = self.predict_proba(X_test)
            # calculate loss for validation set
            current_dataset_idx = 0
            total_validation_loss = 0
            while current_dataset_idx < len(X_test):
                until_idx = min(current_dataset_idx+self.batch_size, len(X_test))
                y_pred = prediction_with_validation[current_dataset_idx:until_idx]
                y_act = y_test[current_dataset_idx:until_idx]

                loss = FFNNClassifier._loss_function(
                    y_act=y_act,
                    y_pred=y_pred,
                    func=self.loss_func
                )
                total_validation_loss += loss * (until_idx - current_dataset_idx)
                current_dataset_idx += self.batch_size
            current_validation_loss = total_validation_loss / len(X_test)

            current_loss = total_loss / len(self.X)
            self.loss_history.append(float(current_loss))
            self.validation_loss_history.append(float(current_validation_loss))
            if self.verbose == 1:
                print(f"Epoch {epoch+1}/{self.epoch_amount} done, Training Loss: {current_loss}, Validation Loss: {current_validation_loss}")

            elif self.verbose == 2:
                print(f"========================================")
                print(f"Epoch {epoch+1}/{self.epoch_amount} done")
                print(f"weights: {self.weights_history}")
                print(f"biases: {self.biases_history}")

        return self.loss_history, self.validation_loss_history

    @staticmethod
    def preprocess_x(X):
        def normalize(X, max_val = 255):
            return X/max_val
        return normalize(X)
    
    @staticmethod
    def preprocess_y(y):
        def one_hot_encode(y, num_of_classes = 10):
            arr = np.zeros((len(y), num_of_classes), dtype="int32")
            for i in range(len(y)):
                arr[i][int(y[i])] = 1
            return arr
        return one_hot_encode(y)
    
    @staticmethod
    def preprocess(X, y):
        return FFNNClassifier.preprocess_x(X), FFNNClassifier.preprocess_y(y)
           

    def predict(self, X_test: NDArray):
        proba = self.predict_proba(X_test)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X_test: NDArray):
        prediction = np.zeros((len(X_test), self._get_amount_of_classes()))
        current_idx = 0
        while current_idx < len(X_test):
            weights, nodes, nodes_active, biases = self._generate_new_empty_layers()
            until_idx = min(current_idx+self.batch_size, len(X_test))
            nodes[0] = X_test[current_idx:until_idx]
            nodes_active[0] = X_test[current_idx:until_idx]

            for k in range(1, len(self.weights_history)+1):
                w_k = self.weights_history[k-1]
                b_k = self.biases_history[k-1]
                h_k_min_1 = nodes_active[k-1]

                a_k = b_k + np.dot(h_k_min_1, w_k)

                nodes[k] = a_k
                nodes_active[k] = FFNNClassifier._activation_function(a_k, self.activation_func[k-1])
            
            prediction[current_idx:until_idx] = nodes_active[-1]
            current_idx += self.batch_size

        return prediction
    

    
    def save(self, filename: str) -> None:
        data = {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation_func": self.activation_func,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "max_epoch": self.epoch_amount,
            "batch_size": self.batch_size,
            "loss_func": self.loss_func,
            "init_method": self.init_method,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "mean": self.mean,
            "std": self.std,
            "seed": self.seed,
            "weights_history": self.weights_history,
            "biases_history": self.biases_history,
            "weight_gradients_history": self.weight_gradients_history,
            "amount_of_features": self.amount_of_features,
            "amount_of_classes": self.amount_of_classes,
            "loss_history": self.loss_history,
            "validation_loss_history": self.validation_loss_history,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        if self.verbose:
            print(f"Model saved to {filename}")

    @staticmethod
    def load(path: str) -> "FFNNClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model = FFNNClassifier(
            hidden_layer_sizes=data["hidden_layer_sizes"],
            activation_func=data["activation_func"],
            learning_rate=data["learning_rate"],
            verbose=data["verbose"],
            max_epoch=data["max_epoch"],
            batch_size=data["batch_size"],
            loss_func=data["loss_func"],
            init_method=data["init_method"],
            lower_bound=data["lower_bound"],
            upper_bound=data["upper_bound"],
            mean=data["mean"],
            std=data["std"],
            seed=data["seed"],
        )
        model.weights_history = data["weights_history"]
        model.biases_history = data["biases_history"]
        model.weight_gradients_history = data["weight_gradients_history"]
        model.amount_of_features = data["amount_of_features"]
        model.amount_of_classes = data["amount_of_classes"]
        model.loss_history = data["loss_history"]
        model.validation_loss_history = data["validation_loss_history"]

        return model