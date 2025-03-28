import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lib.FFNNClassifier import FFNNClassifier
from lib.Utils import load_mnist_dataset, train_test_split, calculate_accuracy, all_element_to_int, download_sample_dataset
import argparse
import time
from lib.NeuralNetworkVisualizer import NeuralNetworkVisualizer
from lib.NeuralNetworkVisualizerPlotly import NeuralNetworkVisualizerPlotly

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="FFNNClassifier")
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "-download",
    nargs=2,
    metavar=("X_PATH", "Y_PATH"),
    help="Download the dataset from mnist and save it to the specified path",
)

group.add_argument(
    '-predict', 
    nargs=4, 
    metavar=('X_PATH', 'Y_PATH', 'UNLABELED_PATH', 'RESULT_PATH'),
    help='Predict using dataset without saving the model'
)

group.add_argument(
    '-save', 
    nargs=3, 
    metavar=('X_PATH', 'Y_PATH', 'MODEL_PATH'),
    help='Train and save the model'
)

group.add_argument(
    '-load',
    nargs=3, 
    metavar=('MODEL_PATH', 'UNLABELED_PATH', 'RESULT_PATH'),
    help='Load and use an existing model'
)

group.add_argument(
    '-accuracy',
    nargs=2, 
    metavar=('PREDICTION_PATH', 'ACTUAL_PATH'),
    help='Get the accuracy of a prediction'
)


group.add_argument(
    '-plot_network',
    nargs=1, 
    metavar=('MODEL_PATH'),
    help='Plot the network'
)
group.add_argument(
    '-plot_weights',
    nargs=1, 
    metavar=('MODEL_PATH'),
    help='Plot the weight distribution'
)
group.add_argument(
    '-plot_gradients',
    nargs=1, 
    metavar=('MODEL_PATH'),
    help='Plot the gradient distribution'
)
group.add_argument(
    '-plot_loss',
    nargs=1,
    metavar=('MODEL_PATH'),
    help='Plot the loss history'
)
parser.add_argument('--layers_to_plot', type=int, nargs='+', default=[0])
parser.add_argument('--plot_size', type=float, default=0.01)

parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[256, 128, 64])
parser.add_argument('--activation_func', nargs='+', default=["sigmoid", "sigmoid", "sigmoid", "softmax"])
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--loss_func', type=str, default="categorical_cross_entropy")
parser.add_argument('--init_method', type=str, default="normal")
parser.add_argument('--lower_bound', type=float, default=5.39294405e-05)
parser.add_argument('--upper_bound', type=float, default=1)
parser.add_argument('--mean', type=float, default=5.39294405e-05)
parser.add_argument('--std', type=float, default=0.44)
parser.add_argument('--seed', type=int, default=69)

parser.add_argument('--l1', type=float, default=0)
parser.add_argument('--l2', type=float, default=0)

args = parser.parse_args()

if args.download:
    X_path, y_path = args.download
    print(f"Downloading:\n  X={X_path}\n  y={y_path}")
    download_sample_dataset(X_path, y_path)
    print("Downloaded!")
elif args.predict:
    X_path, y_path, unlabeled_path, result_path = args.predict
    print(f"Prediction:\n  X={X_path}\n  y={y_path}\n  Unlabeled={unlabeled_path}\n  Result={result_path}")
elif args.save:
    X_path, y_path, model_path = args.save
    print(f"Saving:\n  X={X_path}\n  y={y_path}\n  Model={model_path}")
elif args.load:
    model_path, unlabeled_path, result_path = args.load
    print(f"Loading:\n  Model={model_path}\n  Unlabeled={unlabeled_path}\n  Result={result_path}")
elif args.accuracy:
    prediction_path, actual_path = args.accuracy
    print(f"Accuracy:\n  Prediction={prediction_path}\n  Actual={actual_path}")
elif args.plot_network:
    model_path = args.plot_network[0]
    print(f"Plotting network:\n  Model={model_path}")
elif args.plot_weights:
    model_path = args.plot_weights[0]
    layers_to_plot = args.layers_to_plot
    plot_size = args.plot_size
    print(f"Plotting weights:\n  Model={model_path}\n  Layers={layers_to_plot}\n  Plot Size={plot_size}")
elif args.plot_gradients:
    model_path = args.plot_gradients[0]
    layers_to_plot = args.layers_to_plot
    plot_size = args.plot_size
    print(f"Plotting gradients:\n  Model={model_path}\n  Layers={layers_to_plot}\n  Plot Size={plot_size}")
elif args.plot_loss:
    model_path = args.plot_loss[0]
    print(f"Plotting loss:\n  Model={model_path}")

if args.predict or args.save:
    print("\nTraning Parameters:")
    print("Test Size:", args.test_size)

    print("\nModel Parameters:")
    print("Hidden Layers:", args.hidden_layer_sizes)
    print("Activation Functions:", args.activation_func)
    print("Learning Rate:", args.learning_rate)
    print("Verbose:", args.verbose)
    print("Epochs:", args.max_epoch)
    print("Batch Size:", args.batch_size)
    print("Loss Function:", args.loss_func)
    print("Init Method:", args.init_method)
    print("Lower Bound:", args.lower_bound)
    print("Upper Bound:", args.upper_bound)
    print("Mean:", args.mean)
    print("Std:", args.std)
    print("Seed:", args.seed)
    print("L1:", args.l1)
    print("L2:", args.l2)




# =================== Start Here ===================

def predict_or_save(args, X_path, y_path):

    X, y = load_mnist_dataset(X_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
    y_test_original = all_element_to_int(y_test)
    X_train, y_train = FFNNClassifier.preprocess(X_train, y_train)
    X_test, y_test = FFNNClassifier.preprocess(X_test, y_test)

    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print(y_train)
    print(y_test)

    ffnn = FFNNClassifier(
        hidden_layer_sizes=args.hidden_layer_sizes,
        activation_func=args.activation_func,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        loss_func=args.loss_func,
        init_method=args.init_method,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        mean=args.mean,
        std=args.std,
        seed=args.seed,
        l1=args.l1,
        l2=args.l2,
    )

    print("Training model...")
    start_time = time.time()
    training_loss, validation_loss = ffnn.fit(X_train, y_train, X_test, y_test)
    print("Training done in", f"{time.time() - start_time:.2f}", "seconds")

    print("Predicting to calculate accuracy...")
    prediction = ffnn.predict(X_test)
    print("Accuracy:", calculate_accuracy(prediction, y_test_original) * 100, "%")


    return ffnn

def get_visualizer(ffnn: FFNNClassifier):
    layers = ffnn._get_hidden_layer_sizes()
    weights = ffnn.weights_history
    biases = ffnn.biases_history
    weight_gradients = ffnn.weight_gradients_history
    loss_history = ffnn.loss_history
    nnv = NeuralNetworkVisualizerPlotly(
        layers=layers,
        weights=weights,
        gradients=weight_gradients,
        biases=biases,
        loss_history=loss_history,
    )
    return nnv

if args.predict:
    X_path, y_path, unlabeled_path, result_path = args.predict
    ffnn = predict_or_save(args, X_path, y_path)

    print("Reading unlabeled dataset...")
    X_unlabeled = pd.read_csv(unlabeled_path).to_numpy()
    X_unlabeled = FFNNClassifier.preprocess(X_unlabeled)
    
    print("Predicting...")
    prediction = ffnn.predict(X_unlabeled)
    print("Writing result to CSV...")
    pd.DataFrame(prediction).to_csv(result_path, index=False)
    print("Prediction done!")


if args.save:
    X_path, y_path, model_path = args.save
    ffnn = predict_or_save(args, X_path, y_path)

    print("Saving model...")
    ffnn.save(model_path)
    print("Model saved!")

if args.load:
    model_path, unlabeled_path, result_path = args.load

    print("Loading model...")
    ffnn = FFNNClassifier.load(model_path)

    print("Reading unlabeled dataset...")
    X_unlabeled = pd.read_csv(unlabeled_path).to_numpy()
    X_unlabeled = FFNNClassifier.preprocess_x(X_unlabeled)
    
    print("Predicting...")
    prediction = ffnn.predict(X_unlabeled)
    print("Prediction done!")

    print("Writing result to CSV...")
    pd.DataFrame(prediction).to_csv(result_path, index=False)
    print("Result saved!")

if args.accuracy:
    prediction_path, actual_path = args.accuracy

    print("Reading prediction and actual dataset...")
    prediction = pd.read_csv(prediction_path).to_numpy()
    actual = pd.read_csv(actual_path).to_numpy()

    print("Calculating accuracy...")
    accuracy = calculate_accuracy(prediction, actual)
    print("Accuracy:", accuracy * 100, "%")


if args.plot_network:
    model_path = args.plot_network[0]
    ffnn = FFNNClassifier.load(model_path)
    nnv = get_visualizer(ffnn)
    nnv.plot_network()

if args.plot_weights:
    model_path = args.plot_weights[0]
    layers_to_plot = args.layers_to_plot
    plot_size = args.plot_size
    ffnn = FFNNClassifier.load(model_path)
    nnv = get_visualizer(ffnn)
    nnv.plot_weight_distribution(layers_to_plot, plot_size)

if args.plot_gradients:
    model_path = args.plot_gradients[0]
    layers_to_plot = args.layers_to_plot
    plot_size = args.plot_size
    ffnn = FFNNClassifier.load(model_path)
    nnv = get_visualizer(ffnn)
    nnv.plot_gradient_distribution(layers_to_plot, plot_size)

if args.plot_loss:
    model_path = args.plot_loss[0]
    ffnn = FFNNClassifier.load(model_path)
    nnv = get_visualizer(ffnn)
    nnv.plot_loss()