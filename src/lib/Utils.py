import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
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