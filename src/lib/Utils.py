import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from lib.FFNNClassifier import FFNNClassifier
from sklearn.datasets import fetch_openml
from matplotlib.axes._axes import Axes

def download_sample_dataset(X_path="dataset/X.csv", y_path="dataset/y.csv"):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    pd.DataFrame(X).to_csv(X_path, index=False)
    pd.DataFrame(y).to_csv(y_path, index=False)

def load_mnist_dataset(X_path="dataset/X.csv", y_path="dataset/y.csv"):
    X_csv = None
    y_csv = None
    try:
        print("Reading dataset...")
        X_csv = pd.read_csv(X_path)
        y_csv = pd.read_csv(y_path)
    except:
        print("Dataset Not found! Downloading dataset...")
        download_sample_dataset(X_path, y_path)
        print("Reading dataset...")
        X_csv = pd.read_csv("dataset/X.csv")
        y_csv = pd.read_csv("dataset/y.csv")

    X_data = X_csv.to_numpy()
    y_data_temp = y_csv.to_numpy()
    y_data = np.zeros(len(y_data_temp))
    for k in range(len(y_data_temp)):
        y_data[k] = y_data_temp[k][0]

    return X_data, y_data

# not random, just first 90% of the data
def train_test_split(X, y, test_size=0.1):
    n = len(y)
    n_train = int(n * (1 - test_size))
    X_train = np.array(X[:n_train])
    y_train = np.array(y[:n_train])
    X_test  = np.array(X[n_train:])
    y_test  = np.array(y[n_train:])
    return X_train, X_test, y_train, y_test

def one_hot_encode(y):
    num_of_classes = 10
    arr = []
    for i in range(len(y)):
        arr.append([0 for j in range(num_of_classes)]) # hardcoded 10 for the number of classes
        arr[i][y[i]] = 1
    return arr

def normalize(X):
    return X / 255

def get_same_count(y1, y2):
    count = 0
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            count += 1
    return count

def calculate_accuracy(y1, y2):
    return get_same_count(y1, y2) / len(y1)

def all_element_to_int(arr):
    return np.array([int(i) for i in arr])



def visualize_dataset(X, y, row_count, col_count, offset = 0):
    # scale = np.abs(X).max()
    scale = 255 # in case we only pick some data and none of them reach the max value (255). 
    fig, axes = plt.subplots(row_count, col_count, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(row_count * col_count):
        ax: Axes = axes[i // col_count, i % col_count]
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(str(i+offset)+": "+str(int(y[i+offset])))
        
        ax.imshow(
            X[i+offset].reshape(28, 28),
            interpolation="nearest",
            cmap=plt.cm.RdBu,
            vmin=-scale,
            vmax=scale,
        )

    plt.show()

def model_scratch_output(custom_mlp: FFNNClassifier, X_train_scaled, y_train_one_hot, X_test_scaled, y_test_one_hot, is_only_show_accuracy: bool = False):
    print("[From Scratch FFNNClassifier]")
    custom_mlp.fit(X_train_scaled, y_train_one_hot, X_test_scaled, y_test_one_hot)
    custom_pred = custom_mlp.predict(X_test_scaled)
    custom_pred_proba = custom_mlp.predict_proba(X_test_scaled)
    y_test_labels = np.argmax(y_test_one_hot, axis=1)
    custom_accuracy = accuracy_score(y_test_labels, custom_pred)
    if is_only_show_accuracy:
        print("Accuracy:\n", custom_accuracy)
    else:
        print("Weights:\n", custom_mlp.weights_history)
        print("Biases:\n", custom_mlp.biases_history)
        print("Prediction:\n", custom_pred)
        print("Prediction Probability:\n", custom_pred_proba)
        print("Loss:\n", custom_mlp.loss_history)
        print("Accuracy:\n", custom_accuracy)


def model_comparison(sk_mlp: MLPClassifier, custom_mlp: FFNNClassifier, X_train_scaled, y_train, y_train_one_hot, X_test_scaled, y_test, y_test_one_hot, is_only_show_accuracy: bool = False):
    # print(X_train_scaled)
    # print(y_train)
    # print(y_train_one_hot)
    # print(X_test_scaled)
    # print(y_test)
    # print(y_test_one_hot)

    print("[SKLearn MLPClassifier]")
    sk_mlp.fit(X_train_scaled, y_train)
    sk_pred = sk_mlp.predict(X_test_scaled)
    sk_pred_proba = sk_mlp.predict_proba(X_test_scaled)
    sk_accuracy = accuracy_score(y_test, sk_pred)
    if is_only_show_accuracy:
        print("Accuracy:\n", sk_accuracy)
    else:
        print("Weights:\n", sk_mlp.coefs_)
        print("Biases:\n", sk_mlp.intercepts_)
        print("Prediction:\n", sk_pred)
        print("Prediction Probability:\n", sk_pred_proba)
        print("Loss:\n", sk_mlp.loss_curve_)
        print("Accuracy:\n", sk_accuracy)
    print()

    print("[From Scratch FFNNClassifier]")
    custom_mlp.fit(X_train_scaled, y_train_one_hot, X_test_scaled, y_test_one_hot)
    custom_pred = custom_mlp.predict(X_test_scaled)
    custom_pred_proba = custom_mlp.predict_proba(X_test_scaled)
    y_test_labels = np.argmax(y_test_one_hot, axis=1)
    custom_accuracy = accuracy_score(y_test_labels, custom_pred)
    if is_only_show_accuracy:
        print("Accuracy:\n", custom_accuracy)
    else:
        print("Weights:\n", custom_mlp.weights_history)
        print("Biases:\n", custom_mlp.biases_history)
        print("Prediction:\n", custom_pred)
        print("Prediction Probability:\n", custom_pred_proba)
        print("Loss:\n", custom_mlp.loss_history)
        print("Accuracy:\n", custom_accuracy)
    print()

    print("[Comparison Result]")
    if(is_arr_equal(sk_mlp.coefs_, custom_mlp.weights_history)): print("✅ Weight is equal")
    else: print("❌ Weight is not equal")
    if(is_arr_equal(sk_mlp.intercepts_, custom_mlp.biases_history)): print("✅ Bias is equal")
    else: print("❌ Bias is not equal")
    if(is_arr_equal(sk_pred, custom_pred)): print("✅ Prediction is equal")
    else: print("❌ Prediction is not equal")
    if(is_arr_equal(sk_pred_proba, custom_pred_proba)): print("✅ Prediction Probability is equal")
    else: print("❌ Prediction Probability is not equal")
    if(is_arr_equal(sk_mlp.loss_curve_, custom_mlp.loss_history)): print("✅ Loss is equal")
    else: print("❌ Loss is not equal")
    if(is_arr_equal(sk_accuracy, custom_accuracy)): print("✅ Accuracy is equal")
    else: print("❌ Accuracy is not equal")
    print()

def is_arr_equal(arr1, arr2):
    # numpy check if array of float is equal with tolerance
    if isinstance(arr1, list) and isinstance(arr2, list) or isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        for i in range(len(arr1)):
            if not np.allclose(arr1[i], arr2[i], rtol=1e-03, atol=1e-06):
                print(arr1[i], "!=", arr2[i])
                return False
        return True
    else:
        return np.allclose(arr1, arr2, rtol=1e-05, atol=1e-08)