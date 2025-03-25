from typing import Literal, Callable, Union, List
import numpy as np
from numpy.typing import NDArray, ArrayLike
from lib.WeightInitialization import WeightInitiator
import pickle


class FFNNClassifier:
    def __init__(self,
            hidden_layer_sizes: NDArray,
            activation_func: List[Literal['linear', 'relu', 'sigmoid', 'tanh', 'softmax']],
            learning_rate: float,
            verbose: int, # 0: no print, 1: print epoch progress
            max_epoch: int,
            batch_size: int,
            loss_func: Literal['mean_squared_error', 'binary_cross_entropy', 'categorical_cross_entropy'],
            init_method: Literal['normal', 'zero', 'uniform'] = 'zero',
            lower_bound: float = 0.0,
            upper_bound: float = 1.0,
            mean: float = 0.0,
            std: float = 1.0,
            seed: int | None = None,
        ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.X: NDArray = []
        self.y: list[ArrayLike] = []
        self.weights_history: list[NDArray] = [] # array of weight matrix. index is current epoch
        self.biases_history: list[ArrayLike] = [] # array of bias list. index is current epoch
        self.weight_gradients_history: list[NDArray] = [] # array of weight gradients. index is current epoch

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


    # return [ matrix, matrix, matrix ... ] where matrix is the weight adjacency matrix for each layer. length should be number of layers - 1 because its like the edges/connection between the nodes
    def _generate_initial_weights(self):
        len_features = self._get_amount_of_features()
        layers = np.copy([len_features])
        len_classes = np.array([self._get_number_of_classes()])
        layers = np.append(layers, self.hidden_layer_sizes)
        layers = np.append(layers, len_classes)
        # print("layers:", layers)
        self.init =  WeightInitiator( 
            init_method=self.init_method,
            nodes=layers,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            mean=self.mean,
            std=self.std,
            seed=self.seed
        )
        return self.init.get_weights()

    # return [ float, float, float ... ] where float is the bias for each layer. length should be number of layers - 1 because input layer does not have bias
    def _generate_initial_biases(self):
        bias = self.init.get_bias()
        return bias


# region functions
    @staticmethod
    def _activation_function(x: Union[float, NDArray], func: str):
        if func == 'linear': return x
        elif func == 'relu': return np.maximum(0, x)
        elif func == 'sigmoid': return 1.0/(1.0 + np.exp(-x))
        elif func == 'tanh': return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True) # keepdims=True will keep the dimension of the original array
        raise "Activation function not supported!"

    @staticmethod
    def _activation_derived_function(x: Union[float, NDArray], func: str):
        if func == 'linear': return np.ones_like(x)
        elif func == 'relu': return np.where(x > 0, 1, 0)
        elif func == 'sigmoid':
            sig = FFNNClassifier._activation_function(x, 'sigmoid')
            return sig * (1 - sig)
        elif func == 'tanh':
            p = 2.0/(np.exp(x) - np.exp(-x)) # should be the same as 1 - np.tanh(x) ** 2. will check later
            return p*p  
        elif func == 'softmax':
            batch_size, n = x.shape
            jacobians = np.zeros((batch_size, n, n))

            for i in range(batch_size):
                s = x[i].reshape(-1, 1)
                jacobians[i] = np.diagflat(s) - s @ s.T

            return jacobians
        raise "Activation function not supported!"


    @staticmethod
    def _loss_function(y_act, y_pred, func: str):
        if func == 'mean_squared_error': 
            return 2/len(y_act[0]) * (y_pred - y_act) # 2/len(y_act[0]) can actually be ignored because we have learning rate
        elif func == 'binary_cross_entropy': 
            return (y_pred - y_act)/(y_pred*(1 - y_pred)) * len(y_act[0]) # len(y_act[0]) can actually be ignored because we have learning rate
        elif func == 'categorical_cross_entropy': 
            return (y_pred - y_act) / len(y_act[0]) # len(y_act[0]) can actually be ignored because we have learning rate
        


# endregion functions


# region getters setters

    def _get_amount_of_features(self) -> int:
        if self.amount_of_features == -1: raise Exception("X and y is not set yet")
        return self.amount_of_features

    # Can only be called after setting X and y
    def _get_hidden_layer_sizes(self) -> np.typing.NDArray:

        len_features = self._get_amount_of_features()
        len_classes = self._get_number_of_classes()
        layer_sizes = np.zeros(len(self.hidden_layer_sizes)+2, dtype=int)
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
            weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1])))
            biases.append(np.zeros(layer_sizes[i+1]))
        for i in range(network_depth):
            nodes.append(np.zeros(layer_sizes[i]))
            nodes_active.append(np.zeros(layer_sizes[i]))

        return weights, nodes, nodes_active, biases


    # Can only be called after setting X and y
    def _get_number_of_classes(self):
        # return len(np.unique(self.y))
        return 10 # hardcoded because it messes up things when the possible value is not much

    def copy_list_as_zeros(self, list: list[NDArray]):
        return [np.zeros_like(w) for w in list]

# endregion getters setters



    def fit(self, X: NDArray, y: NDArray):
        if type(y) == list and type(y[0]) != list:
            y = np.array([[i] for i in y])
        if type(y) == list and type(y[0]) == list:
            y = np.array(y)

        if len(X) != len(y):
            raise Exception("length of X and y is not the same")
        if len(X) == 0:
            raise Exception("len(self.X) == 0")
        if len(X[0]) == 0:
            raise Exception("len(self.X[0]) == 0")
        if len(y) == 0:
            raise Exception("len(self.y) == 0")
        
        self.amount_of_features = len(X[0])

        # clean up in case this function is called multiple times
        self.X: NDArray = []
        self.y: ArrayLike = []
        self.weights_history: list[NDArray] = []
        self.biases_history: list[ArrayLike] = []
        self.weight_gradients_history: list[NDArray] = []


        self.X = X
        self.y = y
        initial_weight = self._generate_initial_weights()
        initial_bias = self._generate_initial_biases()
        initial_gradients = [np.zeros_like(w) for w in initial_weight]
        self.weights_history = initial_weight
        self.biases_history = initial_bias
        self.weight_gradients_history = initial_gradients

        layer_sizes = self._get_hidden_layer_sizes()
        network_depth = len(layer_sizes)
        number_of_classes = self._get_number_of_classes()

        for epoch in range(self.epoch_amount):
            # for current_dataset_idx in range(len(self.X)):
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

                    a_k = b_k + np.dot(h_k_min_1, w_k) # numpy will automatically broadcast b_k (row will be copied to match the result from dot) so that this is addable

                    nodes[k] = a_k
                    nodes_active[k] = FFNNClassifier._activation_function(a_k, self.activation_func[k-1])

                # print([p.shape for p in nodes_active])
                loss_grad = FFNNClassifier._loss_function(
                    y_act=self.y[current_dataset_idx:until_idx],
                    y_pred=nodes_active[network_depth-1],
                    func=self.loss_func
                )
                # print("self.y[current_dataset_idx:until_idx]: ", self.y[current_dataset_idx:until_idx])
                # print("nodes_active[network_depth-1]: ", nodes_active[network_depth-1])
                # print("loss_grad: ", loss_grad)


                # Backward Propagation
                weight_gradiens = [0 for i in range(len(self.weights_history))] # 0 will be replaced with numpy.array
                bias_gradiens = [0 for i in range(len(self.biases_history))] # 0 will be replaced with numpy.array


                delta = 0
                if self.activation_func[-1] == 'softmax' and self.loss_func == 'categorical_cross_entropy':
                    delta = loss_grad # already simplified (y_pred - y_true)
                elif self.activation_func[-1] == 'softmax' and self.loss_func != 'categorical_cross_entropy':
                    jacobians = FFNNClassifier._activation_derived_function(nodes[-1], self.activation_func[-1])
                    loss_grad_col = loss_grad[..., np.newaxis] # (batch_size, n) -> (batch_size, n, 1)
                    delta = np.matmul(jacobians, loss_grad_col)  # (batch_size, n, 1)
                    delta = np.squeeze(delta, axis=-1) # (batch_size, n)
                else:
                    delta = loss_grad * FFNNClassifier._activation_derived_function(nodes[-1], self.activation_func[-1])


                weight_gradiens[network_depth-2] = np.dot(nodes_active[-2].T, -delta)
                bias_gradiens[network_depth-2] = -delta


                for k in range(network_depth-2, 0, -1): # from the last hidden layer (not including the output layer)
                    w = self.weights_history[k]

                    delta = np.dot(delta, w.T) * FFNNClassifier._activation_derived_function(nodes[k], self.activation_func[k-1])

                    weight_gradiens[k-1] = np.dot(nodes_active[k-1].T, -delta)
                    bias_gradiens[k-1] = -delta

                self.weight_gradients_history = weight_gradiens

                # Update
                for k in range(network_depth-1):
                    w_k = self.weights_history[k]
                    b_k = self.biases_history[k]

                    weights[k] = w_k + self.learning_rate * weight_gradiens[k]

                    biases[k] = b_k + self.learning_rate * bias_gradiens[k]
                self.weights_history = weights
                self.biases_history = biases


                current_dataset_idx += self.batch_size
            #### while loop ends here ##############################################


            if self.verbose == 1:
                print(f"Epoch {epoch+1}/{self.epoch_amount} done")
            elif self.verbose == 2:
                print(f"========================================")
                print(f"Epoch {epoch+1}/{self.epoch_amount} done")
                print(f"weights: {self.weights_history}")
                print(f"biases: {self.biases_history}")


    @staticmethod
    def preprocess_x(X):
        def normalize(X, max_val = 255):
            return X/max_val
        return normalize(X)
    
    @staticmethod
    def preprocess_y(y):
        def one_hot_encode(y, num_of_classes = 10):
            arr = np.zeros((len(y), num_of_classes))
            for i in range(len(y)):
                arr[i][int(y[i])] = 1
            return arr
        return one_hot_encode(y)
    
    @staticmethod
    def preprocess(X, y):
        return FFNNClassifier.preprocess_x(X), FFNNClassifier.preprocess_y(y)
           

    def predict(self, X_test: NDArray):
        prediction = np.zeros(len(X_test), dtype=int)
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
            predicted_class = [int(np.argmax(nodes_active[-1][i])) for i in range(len(nodes_active[-1]))] # idx with highest value. idx is also the class
            prediction[current_idx:until_idx] = predicted_class
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
            "amount_of_features": self.amount_of_features
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

        return model