
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from lib.FFNNClassifier import FFNNClassifier
from lib.Utils import normalize
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
hidden_layer_sizes=[128,64,32]
max_iter=15
init_method="normal"
learning_rate_init=0.01
batch_size=50
activation_mlplib="logistic"
activation_ffnn="sigmoid"

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
    activation_func=[activation_ffnn] * len(hidden_layer_sizes) + ['softmax']
)


X = pd.read_csv("dataset/X.csv").to_numpy()
y = pd.read_csv("dataset/y.csv").to_numpy()
X = X.astype('float32')
y = y.astype('int')
y = y.reshape(-1)

train_until_idx = int(0.8 * 70000)
test_until_idx = int(0.2 * 70000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[:train_until_idx]
y_train = y_train[:train_until_idx]
X_test = X_test[:test_until_idx]
y_test = y_test[:test_until_idx]

scaler = StandardScaler()
X_train_scaled = normalize(X_train)
X_test_scaled = normalize(X_test)
y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

model_comparison(sk_mlp, custom_mlp, X_train_scaled, y_train, y_train_one_hot, X_test_scaled, y_test, y_test_one_hot, True)