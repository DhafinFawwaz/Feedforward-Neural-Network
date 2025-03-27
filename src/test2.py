
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

def one_hot_encode(y, num_classes=10):
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((len(y), num_classes), dtype=int)
    print(y)
    print(len(y))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

print("Loading MNIST dataset...")
# X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = pd.read_csv("dataset/X.csv").to_numpy()
y = pd.read_csv("dataset/y.csv").to_numpy()
X = X.astype('float32')
y = y.astype('int')
y = y.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for i in range(len(y_train)):
    if y_train[i] == 0 and y_train[i+1] == 1 and y_train[i+2] == 2 and y_train[i+3] == 3:
        print(i)
        break
for i in range(len(y_train)):
    if y_test[i] == 0 and y_test[i+1] == 1 and y_test[i+2] == 2 and y_test[i+3] == 3:
        print(i)
        break


from_idx = 18771
until_idx = 18774
test_from_idx = 392
test_until_idx = 395

X_train = X_train[from_idx:until_idx]
y_train = y_train[from_idx:until_idx]
X_test = X_test[test_from_idx:test_until_idx]
y_test = y_test[test_from_idx:test_until_idx]

print("X_train:",X_train)
print("X_test:",X_test)
print("y_train:",y_train)
print("y_test:",y_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_one_hot = one_hot_encode(y_train, 3)
y_test_one_hot = one_hot_encode(y_test, 3)



def model_comparison(sk_mlp: MLPClassifier, custom_mlp: FFNNClassifier, X_train_scaled, y_train, y_train_one_hot, is_weight: bool = False):
    print("Fitting SKLearn...")
    # print(dir(sk_mlp))
    # return
    if not is_weight:
        print("X_train_scaled:",X_train_scaled)
        print("y_train:",y_train)
        sk_mlp.fit(X_train_scaled, y_train)
    # sk_pred = sk_mlp.predict(X_test_scaled)
    # sk_accuracy = accuracy_score(y_test, sk_pred)
    # print("[SKLEARN] Prediction: ",sk_pred)
    # print("[SKLEARN] Accuracy: ", sk_accuracy)
    # print("[SKLEARN] Weights: ", sk_mlp.coefs_)
    # print("[SKLEARN] Bias: ", sk_mlp.intercepts_)
    print(sk_mlp.coefs_[-1])
    print()

    print("Fitting Custom...")
    if not is_weight:
        print("X_train_scaled:",X_train_scaled)
        print("y_train_one_hot:",y_train_one_hot)
        custom_mlp.fit(X_train_scaled, y_train_one_hot)
    # custom_pred = custom_mlp.predict(X_test_scaled)
    # y_test_labels = np.argmax(y_test_one_hot, axis=1)
    # custom_accuracy = accuracy_score(y_test_labels, custom_pred)
    # print("[CUSTOM] Prediction: ",custom_pred)
    # print("[CUSTOM] Accuracy: ", custom_accuracy)
    # print("[SKLEARN] Weights: ", custom_mlp.weights_history)
    # print("[SKLEARN] Bias: ", custom_mlp.biases_history[-1])
    print(custom_mlp.weights_history[-1])
    print()




lower_bound=5.39294405e-05
upper_bound=1
mean=5.39294405e-05
std=.44
seed=69
hidden_layer_sizes=[1]
max_iter=1
init_method="normal"
learning_rate_init=0.01
batch_size=1
activation="relu"

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
    activation=activation,
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
    activation_func=[activation] * len(hidden_layer_sizes) + ['softmax']
)

X_train_scaled = np.array([
    [0.1,0.2,0.3],
    [0.4,0.5,0.6],
    [0.7,0.8,0.9]
])
y_train = np.array([0,1,2])
y_train_one_hot = one_hot_encode(y_train, 3)
model_comparison(sk_mlp, custom_mlp, X_train_scaled, y_train, y_train_one_hot, False)