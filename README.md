# IF3270 Pembelajaran Mesin Feedforward Neural Network

## Setup Project

### Setup Virtual Environment

If using command prompt:
```bash
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```





can only be either one of the following:
-download, -predict, -save, -load, -accuracy

Download the dataset from mnist and save it to the specified path
-download <dataset_filepath_x> <dataset_filepath_y>

Read the dataset and immedietely predict without saving the model
-predict <dataset_filepath_x> <dataset_filepath_y> <unlabeled_filepath> <result_filepath>

Save model
-save <dataset_filepath_x> <dataset_filepath_y> <model_filepath>

Load existing model
-load <model_filepath> <unlabeled_filepath> <result_filepath>

Calculate accuracy
-accuracy <result_filepath> <ground_truth_filepath>

Do this if -predict or -save
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



Example

Download
python main.py -download dataset/X.csv dataset/y.csv

Predict
python main.py -predict dataset/X.csv dataset/y.csv dataset/unlabeled.csv result/result.csv

Save
python main.py -save dataset/X.csv dataset/y.csv model/ffnn_1.pkt

Load
python main.py -load model/ffnn_1.pkt dataset/X_unlabeled.csv result/result.csv

Accuracy
python main.py -accuracy result/result.csv dataset/y.csv

Plot Netowork
python main.py -plot_network model/ffnn_1.pkt

Plot Weight Distribution
python main.py -plot_weights model/ffnn_1.pkt -layers_to_plot 1 -plot_type hist

Plot Gradient Distribution
python main.py -plot_gradients model/ffnn_1.pkt -layers_to_plot 0 1 -plot_type hist
python main.py -plot_weights model/ffnn_2.pkt --layers_to_plot 0 1 2 3 --plot_type line

Save with all parameters
python main.py -save dataset/X.csv dataset/y.csv model/ffnn_1.pkt -test_size 0.1 -hidden_layer_sizes 256 128 64 -activation_func sigmoid sigmoid sigmoid sigmoid -learning_rate 0.05 -verbose 1 -max_epoch 15 -batch_size 50 -loss_func mean_squared_error -init_method normal -lower_bound 5.39294405e-05 -upper_bound 1 -mean 5.39294405e-05 -std .44 -seed 69


Predict
python main.py -predict dataset/X.csv dataset/y.csv dataset/unlabeled.csv result/result.csv --loss_func categorical_cross_entropy --activation_func sigmoid sigmoid sigmoid softmax