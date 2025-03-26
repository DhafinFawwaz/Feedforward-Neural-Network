from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lib.Utils import load_mnist_dataset, train_test_split, calculate_accuracy, all_element_to_int, download_sample_dataset
from lib.FFNNClassifier import FFNNClassifier

X_path = "dataset/X.csv"
y_path = "dataset/y.csv"
X, y = load_mnist_dataset(X_path, y_path)

train_until_idx = 100
X = X[:train_until_idx]
y = y[:train_until_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
y_train_original = all_element_to_int(y_train)
y_test_original = all_element_to_int(y_test)
X_train, y_train = FFNNClassifier.preprocess(X_train, y_train)
X_test, y_test = FFNNClassifier.preprocess(X_test, y_test)

test_until_idx = 50
X_test = X_test[:test_until_idx]
y_test_original = y_test_original[:test_until_idx]

print(X_train.shape)
print(y_train_original.shape)
print(X_test.shape)
print(y_test_original.shape)


print("Starting with MLPClassifier...")
clf = MLPClassifier(
    random_state=1, 
    max_iter=15,
    hidden_layer_sizes=(16, 8, 4),
    batch_size=1
    ).fit(X_train, y_train_original)
prediction = clf.predict(X_test)
print(prediction)
print("Accuracy:", calculate_accuracy(prediction, y_test_original) * 100, "%")



print("Starting with FFNNClassifier...")
ffnn = FFNNClassifier(
    hidden_layer_sizes=(16, 8, 4),
    activation_func=["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
    learning_rate=0.01,
    verbose=False,
    max_epoch=15,
    batch_size=1,
    loss_func="mean_squared_error",
    init_method="normal",
    lower_bound=-1,
    upper_bound=1,
    mean=0,
    std=0.1,
    seed=1
)
ffnn.fit(X_train, y_train)
prediction = ffnn.predict(X_test)
print(prediction)
print("Accuracy:", calculate_accuracy(prediction, y_test_original) * 100, "%")