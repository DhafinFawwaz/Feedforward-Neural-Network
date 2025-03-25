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
-download, -predict, -save, -load

Download the dataset from mnist and save it to the specified path
-download <dataset_filepath_x> <dataset_filepath_y>

Read the dataset and immedietely predict without saving the model
-predict <dataset_filepath_x> <dataset_filepath_y> <unlabeled_filepath> <result_filepath>

Save model
-save <dataset_filepath_x> <dataset_filepath_y> <model_filepath>

Load existing model
-load <model_filepath> <unlabeled_filepath> <result_filepath>

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
