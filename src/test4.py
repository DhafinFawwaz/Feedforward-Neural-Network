
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from lib.FFNNClassifier import FFNNClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from lib.MLPLib import MLPLIB
import random
import pandas as pd

from lib.Utils import model_comparison



def one_hot_encode(y, num_classes=10):
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((len(y), num_classes), dtype=int)
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

lower_bound=5.39294405e-05
upper_bound=1
mean=5.39294405e-05
std=.44
seed=69
hidden_layer_sizes=[4,3,2]
max_iter=3
init_method="normal"
learning_rate_init=0.1
batch_size=3
activation_mlplib="identity"
activation_ffnn="linear"
l1=0.1
l2=0.1

# Scikit-learn MLP
sk_mlp = MLPLIB(
    max_iter=max_iter,
    learning_rate_init=learning_rate_init,
    hidden_layer_sizes=hidden_layer_sizes,
    init_method=init_method,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    mean=mean,
    std=std,
    seed=seed,
    batch_size=batch_size,
    activation=activation_mlplib,
    alpha=l2, # L2 regularization
    alpha_l1=l1, # L1 regularization
)

# Custom MLP
custom_mlp = FFNNClassifier(
    max_epoch=max_iter,
    learning_rate=learning_rate_init,
    hidden_layer_sizes=hidden_layer_sizes,
    init_method=init_method,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    mean=mean,
    std=std,
    seed=seed,
    batch_size=batch_size,
    verbose=1,
    loss_func="categorical_cross_entropy",
    activation_func=[activation_ffnn] * len(hidden_layer_sizes) + ['softmax'],
    l1=l1,
    l2=l2,
)

X_train_scaled = np.array([
    [0.9,0.05,0.35],
    [0.13,0.89,0.05],
    [0.15,0.1,0.9]
])
y_train = np.array([0,1,2])
y_train_one_hot = one_hot_encode(y_train, 3)
X_test_scaled = np.array([
    [0,0.8,0],
    [1,0,0],
    [0,0.1,0.9],

    [0.1,0.2,0.7],
    [0.8,0.1,0.8],
    [0.7,0.2,0.7],
])
y_test = np.array([1,2,0, 2,1,0])
y_test_one_hot = one_hot_encode(y_test, 3)

model_comparison(sk_mlp, custom_mlp, X_train_scaled, y_train, y_train_one_hot, X_test_scaled, y_test, y_test_one_hot)