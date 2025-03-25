import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lib.FFNNClassifier import FFNNClassifier
from lib.Utils import load_mnist_dataset, train_test_split, calculate_accuracy, all_element_to_int

import argparse

parser = argparse.ArgumentParser()

# run_type:
# save, load

# if save, use --predict, can immedietely predict
# if load, immedietely predict and write the result to csv to file
# default is save

# test_size, default 0.1

parser.add_argument('--load', action='store_true', help='Load existing model')
parser.add_argument('--save', action='store_true', help='Save model after training')


parser.add_argument('-sizes', type=int, nargs='+', help='List of sizes')



# Accept multiple values for -sizes
parser.add_argument('-sizes', type=int, nargs='+', help='List of sizes')
parser.add_argument('-learning_rate', type=float, help='Learning rate')
parser.add_argument('-mean', type=float, help='Mean value')

args = parser.parse_args()

print("Sizes:", args.sizes)
print("Learning rate:", args.learning_rate)
print("Mean:", args.mean)

exit(0)

X, y = load_mnist_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_test_original = all_element_to_int(y_test)
X_train, y_train = FFNNClassifier.preprocess(X_train, y_train)
X_test, y_test = FFNNClassifier.preprocess(X_test, y_test)

ffnn = FFNNClassifier(
    hidden_layer_sizes=[256, 128, 64],
    activation_func=["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
    learning_rate=0.05,
    verbose=1,
    max_epoch=15,
    batch_size=50,
    loss_func="mean_squared_error",
    init_method="normal",
    lower_bound=5.39294405e-05,
    upper_bound=1,
    mean=5.39294405e-05,
    std=.44,
    seed=69
)

ffnn.fit(X_train, y_train)
prediction = ffnn.predict(X_test)

print("Accuracy:", calculate_accuracy(prediction, y_test_original) * 100, "%")

print("Saving model...")
ffnn.save("model/ffnn_model.ffnn")