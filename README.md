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





['__abstractmethods__', '__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__firstlineno__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__sklearn_clone__', '__sklearn_tags__', '__static_attributes__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_backprop', '_build_request_for_signature', '_check_feature_names', '_check_n_features', '_check_solver', '_compute_loss_grad', '_doc_link_module', '_doc_link_template', '_doc_link_url_param_generator', '_estimator_type', '_fit', '_fit_lbfgs', '_fit_stochastic', '_forward_pass', '_forward_pass_fast', '_get_default_requests', '_get_doc_link', '_get_metadata_request', '_get_param_names', '_get_tags', '_init_coef', '_initialize', '_loss_grad_lbfgs', '_more_tags', '_parameter_constraints', '_predict', '_repr_html_', '_repr_html_inner', '_repr_mimebundle_', '_score', '_score_with_function', '_unpack', '_update_no_improvement_count', '_validate_data', '_validate_input', '_validate_params', 'activation', 'alpha', 'batch_size', 'beta_1', 'beta_2', 'early_stopping', 'epsilon', 'fit', 'get_metadata_routing', 'get_params', 'hidden_layer_sizes', 'init_method', 'learning_rate', 'learning_rate_init', 'loss', 'lower_bound', 'max_fun', 'max_iter', 'mean', 'momentum', 'n_iter_no_change', 'nesterovs_momentum', 'partial_fit', 'power_t', 'predict', 'predict_log_proba', 'predict_proba', 'random_state', 'score', 'seed', 'set_params', 'set_partial_fit_request', 'set_score_request', 'shuffle', 'solver', 'std', 'tol', 'upper_bound', 'validation_fraction', 'verbose', 'warm_start']