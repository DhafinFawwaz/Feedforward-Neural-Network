# IF3270 Pembelajaran Mesin Feedforward Neural Network

## Memberss
| Name | NIM | Description |
| --- | --- | --- |
| 13522014 | Raden Rafly Hanggaraksa Budiarto | Metode inisialisasi bobot, Perbandingan performa model dengan library, Menyusun laporan |
| 13522084 | Dhafin Fawwaz Ikramullah | Forward propagation, Backward propagation, Debugging dan emastikan output sama dengan MLPClassifier sklearn, Menyusun laporan |
| 13522092 | Sa’ad Abdul Hakim | Visualisasi, Menyusun laporan |

## Description
This project is an implementation of a feedforward neural network from scratch. It has same result with sklearn's MLPClassifier and might have more detailed result because of using float64 instead of float32. 
- `main.py` is the main file to run the project
- `test3.py` is the file to test the implementation with the MNIST dataset
- `test4.py` is the file to test the implementation with manually created dataset with smaller size

You can save the trained model in the `model` folder and load it later to predict new data. You can also calculate the accuracy of the prediction. You can also store the dataset in csv format in the `dataset` folder. The lib folder contains the implementation of the feedforward neural network.
- `FFNNClassidier.py` is the implementation of the feedforward neural network from scratch
- `MLPLib.py` is sklearn's MLPClassifier with overridden initial weights
- `NeuralNetworkVisualizer.py` is a class to visualize the neural network
- `Utils.py` contains utility functions



## Setup Project

### Setup Virtual Environment

If using command prompt:
```bash
cd src
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade nbformat
```

### Quick Command to Run
```bash
python main.py -download dataset/X.csv dataset/y.csv
python main.py -save dataset/X.csv dataset/y.csv model/ffnn_1.pkt
python main.py -plot_network model/ffnn_1.pkt
python main.py -plot_gradients model/ffnn_1.pkt
python main.py -plot_weights model/ffnn_1.pkt
```

### Quick Command to Predict
After downloading the `dataset/X.csv` and saved `model/ffnn_1.pkt`, rename it to `dataset/X_unlabeled.csv` and run the following command
```bash
python main.py -load model/ffnn_1.pkt dataset/X_unlabeled.csv result/result.csv
```


### Useful other Commands for different parameters
Before we start, you can of course just run it with wrong parameters and it will show the correct parameters to use

Download the dataset from mnist and save it to the specified path
```
python main.py -download <dataset_filepath_x> <dataset_filepath_y>
```

Read the dataset and immedietely predict without saving the model
```
python main.py -predict <dataset_filepath_x> <dataset_filepath_y> <unlabeled_filepath> <result_filepath>
```

Save model
```
python main.py -save <dataset_filepath_x> <dataset_filepath_y> <model_filepath>
```

Load existing model
```
python main.py -load <model_filepath> <unlabeled_filepath> <result_filepath>
```

Calculate accuracy
```
python main.py -accuracy <result_filepath> <ground_truth_filepath>
```

Parameters for -predict or -save
```
Set values for -save and -predict. defaults are:
test_size=0.1
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
```


Examples

Download
```
python main.py -download dataset/X.csv dataset/y.csv
```

Predict
```
python main.py -predict dataset/X.csv dataset/y.csv dataset/unlabeled.csv result/result.csv
```

Save
```
python main.py -save dataset/X.csv dataset/y.csv model/ffnn_1.pkt
```

Load
```
python main.py -load model/ffnn_1.pkt dataset/X_unlabeled.csv result/result.csv
```

Accuracy
```
python main.py -accuracy result/result.csv dataset/y.csv
```

Plot Netowork
```
python main.py -plot_network model/ffnn_1.pkt --hidden_layer_sizes 64 32 16
```

Plot Weight Distribution
```
python main.py -plot_weights model/ffnn_1.pkt --plot_size 0.01
```

Plot Gradient Distribution
```
python main.py -plot_gradients model/ffnn_1.pkt --plot_size 0.0001
```

Save with all parameters
```
python main.py -save dataset/X.csv dataset/y.csv model/ffnn_4.pkt -test_size 0.1 -hidden_layer_sizes 256 128 64 -activation_func sigmoid sigmoid sigmoid sigmoid -learning_rate 0.05 -verbose 1 -max_epoch 15 -batch_size 50 -loss_func mean_squared_error -init_method normal -lower_bound 5.39294405e-05 -upper_bound 1 -mean 5.39294405e-05 -std .44 -seed 69
```


Predict with custom activation function and loss function
```
python main.py -predict dataset/X.csv dataset/y.csv dataset/unlabeled.csv result/result.csv --loss_func categorical_cross_entropy --activation_func sigmoid sigmoid sigmoid softmax
```

