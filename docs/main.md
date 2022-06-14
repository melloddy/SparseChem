# Installation

SparseChem depends on **pytorch**, which you have to install first, other dependencies will be installed with the package:

```
pip install -e .
```

# ChEMBL Example
First data has to be downloaded into `examples/chembl`:
```
https://www.esat.kuleuven.be/~aarany/chembl_23_x.mtx
https://www.esat.kuleuven.be/~aarany/chembl_23_y.mtx
https://www.esat.kuleuven.be/~aarany/folding_hier_0.6.npy
```

Then execute training:
```
cd ./examples/chembl
python train.py
```

## Specifying parameters of the network
Single layer network with `400` hidden:
```
python train.py \
  --x ./chembl_23_x.mtx \
  --y ./chembl_23_y.mtx \
  --folding ./folding_hier_0.6.npy \
  --fold_va 0 \
  --batch_ratio    0.02 \
  --hidden_sizes   400 \
  --last_dropout   0.2 \
  --middle_dropout 0.2 \
  --weight_decay   0.0 \
  --epochs         20 \
  --lr             1e-3 \
  --lr_steps       10 \
  --lr_alpha       0.3
```
We use `0.2` dropout and with no weight decay (regularization).
The total of epochs is `20` and learning rate is `1e-3`.
We also add `--lr_steps 10` that means that the after 10 epochs the learning rate is multiplied by 0.3 (`lr_alpha` value).

This should get us to 0.83 average AUC for tasks with 25 positives and 25 negatives.

## Two layer network
To get a two layer network we just add several values to `--hidden_sizes`.
```
python train.py \
  --x ./chembl_23_x.mtx \
  --y_class ./chembl_23_y.mtx \
  --folding ./folding_hier_0.6.npy \
  --fold_va 0 \
  --batch_ratio    0.02 \
  --hidden_sizes   400 400 \
  --weight_decay   1e-4 \
  --last_dropout   0.2 \
  --middle_dropout 0.2 \
  --epochs         20 \
  --lr             1e-3 \
  --lr_steps       10 \
  --lr_alpha       0.3
```
We also modified the weight decay to `1e-4`.

## AUC calculation
The script uses all data for training but AUCs are calculated only on tasks with enough positive and negative examples, default is `25` each.
To instead require at least 50 positives and 50 negatives, add `--min_samples_auc 50`.

There are few options to reduce the time spent for AUC calculations:
* `--eval_train 0` will turn off AUC calculation for the training set.
* `--eval_frequency 2` will specify that AUCs should be calculated only every 2 epochs (default is 1). If set to `-1` the evaluation is only done once at the end of the run.

## Input folding
The pipeline also provides an option to fold inputs to a smaller size.
For example, adding `--fold_inputs 20000` folds the inputs to 20,000 dimension.
This is useful for reducing the model size, without hurting the performance too much.

## Task weighting
SparseChem also supports task weighting.
This can be enabled by adding a `--weights_class weights_c.csv` option for classification and with `--weights_regr weights_r.csv` for regression.
The CSV file can contain the following columns:
* `task_id` integer from 0 to number of classification tasks minus 1,
* `training_weight` real value between 0.0 and 1.0 (inclusive) for each task,
* `censored_weight` (optional) real value between 0.0 and 1.0 (inclusive). It allows per task down-weighting of censored values in **regression**.
* `aggregation_weight` (optional) real value between 0.0 and 1.0 (inclusive). If specified determines weight when **aggregating metrics** (AUC-PR etc).
* `task_type` (optional) string specifying task type.

The number of tasks in the CSV file `--weights_class` (`--weights_regr`) must be equal to the number of tasks in `--y_class` matrix (`--y_regr`).

## Regression
SparseChem also supports regression and also both regression and classification jointly.
Here is an example to use regression:
```
python train.py \
  --x ./chembl_23_x.mtx \
  --y_regr ./chembl_23_y.mtx \
  --folding ./folding_hier_0.6.npy \
  --fold_va 0 \
  --batch_ratio    0.02 \
  --hidden_sizes   400 400 \
  --last_non_linearity tanh \
  --weight_decay   1e-4 \
  --last_dropout   0.2 \
  --middle_dropout 0.2 \
  --epochs         20 \
  --lr             1e-3 \
  --lr_steps       10 \
  --lr_alpha       0.3
```
We matrix for `--y_regr` is sparse matrix (similar to classification).
For which SparseChem minimizes the mean squared error (MSE) loss.
Note we have also switched the non-linearity to `tanh`.

## Censored regression
It is possible to use censored regression by passing an extra sparse matrix with `--y_censor chembl_censor.npy` that has the same sparsity pattern as `--y_regr`.
The censor matrix should store the censoring mask for each regression value:
* `-1` for below censoring
* `0` for no censoring (usual regression)
* `+1` for upper censoring

## Running on CPU or other GPUs
The default device is `cuda:0`.
To train the model on CPU just add `--dev cpu` to the arguments.
Similarly, to choose another GPU, we can specify `--dev cuda:1`.

## Predicting on new compounds
After the run is complete the model's **parameters** and **conf** are saved under `models/` folder.
Note you can change the output directory by providing `--output_dir some_other_dir`.

We then can use `predict.py` to make predictions for new compounds as follows:
```bash
python predict.py \
    --x new_compounds.mtx \
    --outprefix y_hat \
    --conf models/sc_chembl_h400.400_ldo0.2_wd1e-05.json \
    --model models/sc_chembl_h400.400_ldo0.2_wd1e-05.pt \
    --dev cuda:0
```
where `new_compounds.mtx` is the sparse feature matrix of the new compounds and `--outprefix y_hat` specifies the prefix for the file(s) where the predictions are saved to.
In this example, the output file for **classification** tasks will be `y_hat-class.npy` and for **regression** tasks `y_hat-regr.npy`,
assuming that the model has both types of tasks.
The `--conf` and `--model` should point to the configuration and model files that where saved during the training.

The format for the prediction is a Numpy file that can be loaded as follows:
```python
import numpy as np
y_hat = np.load("y_hat.npy")
```
The predictions themselves are class probabilities (values between 0.0 and 1.0).

There is an option `--dropout 1` to switch on the dropout during predictions to obtain stochastic predictions, *e.g.*, for MC-dropout. 

## Sparse predictions
It is also possible to predict only **selected elements** instead of the whole output matrix (either classification and/or regression):
* add `--y_class y_class_to_predict.npy` (.npy, .npz or .mtx) for classification,
* add `--y_regr y_regr_to_predict.npy` (.npy, .npz or .mtx) for regression.
If added, these matrices specify locations where to make predictions to.

Here is an example for classification:
```bash
python predict.py \
    --x new_compounds.mtx \
    --y_class y_to_predict.npy \
    --outprefix y_hat \
    --conf models/sc_chembl_h400.400_ldo0.2_wd1e-05.json \
    --model models/sc_chembl_h400.400_ldo0.2_wd1e-05.pt \
    --dev cuda:0
```
Then `predict.py` will create a file `y_hat-class.npy` which contains now `scipy.sparse` matrix with the same shape and the same sparsity pattern as `y_to_predict.npy`.
The resulting file can be loaded by `yhat = np.load('y_hat-class.npy', allow_pickle=True).item()` or just by:
```python
import sparsechem as sc
yhat = sc.load_sparse('y_hat-class.npy')
```

### Sparse predictions for specific folds

Additionally, if only a specific fold is of interest out of Y then we can add parameters `--folding my_folding.npy` and `--predict_fold 0` to only return predictions for rows who are in **fold 0** (where the folding of samples is specified by `my_folding.npy`).
It is possible to predict several folds by specifying them as `--predict_fold 0 2 3`.
Note that the rows from the other (unspecified) folds will be just empty in the `y-hat-class.npy` matrix.

## Retreiving last hidden layers
Instead of outputting the predictions we can use `predict.py` to output the activations of the last layer.
This can be done by adding the option `--last_hidden 1`.

Then the output file will contain the numpy matrix of the hidden vectors, which can be loaded the same way as predictions.


## Full list of all command line arguments

* __--x__: Activity file (matrix market, .npy or .npz) (str)
* __--y_class | --y | --y_classification__: Activity file (matrix market, .npy or .npz) (str)
* __--y_regr | --y_regression__: Activity file (matrix market, .npy or .npz) (str)
* __--y_censor__: Censor mask for regression (matrix market, .npy or .npz) (str)
* __--weights_class | --task_weights | --weights_classification__: CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks) (str)
* __--weights_regr | --weights_regression__: CSV file with columns task_id, training_weight, censored_weight, aggregation_weight, aggregation_weight, task_type (for regression tasks) (str)
* __--censored_loss__: Set this to 0 if censored loss should not be used for training (int, default=1)
* __--folding__: Folding file (npy) (str)
* __--fold_va__: Validation fold number (int, default=0)
* __--fold_te__: Test fold number (removed from dataset) (int)
* __--batch_ratio__: Batch ratio (float, default=0.02)
* __--internal_batch_max__: Maximum size of the internal batch (int)
* __--normalize_loss__: Normalization constant to divide the loss (int, default=None,  uses batch size)
* __--normalize_regression__: Set this to 1 if the regression tasks should be normalized
* __--hidden_sizes__: Hidden sizes (int) (default: [])
* __--last_hidden_sizes__: Hidden sizes in the head (int) (default: [])
* __--middle_dropout__: Dropout for layers before the last (float, default=0.0)
* __--last_dropout__: Last dropout (float, default=0.2)
* __--weight_decay__: Weight decay (float, default=0.0)
* __--last_non_linearity__: Last layer non-linearity (str, default="relu", choices=["relu", "tanh"])
* __--non_linearity__: Before last layer non-linearity (str, default="relu", choices=["relu", "tanh"])
* __--input_transform__: Transformation to apply to inputs (str, default="binarize", choices=["binarize", "none", "tanh"])
* __--lr__: Learning rate (float, default=1e-3)
* __--lr_alpha__: Learning rate decay multiplier (float, default=0.3)
* __--lr_steps__: Learning rate decay steps (int, default=[10])
* __--input_size_freq__: Number of high importance features (int)
* __--fold_inputs__: Fold input to a fixed set (default no folding) (int)
* __--epochs__: Number of epochs (type=int, default=20)
* __--min_samples_auc__: Minimum number samples (in each class) for AUC calculation" (int, default=25)
* __--min_samples_class__: Minimum number samples in each **class** and in each **fold** for AUC calculation (only used if aggregation_weight is not provided in --weights_class) (int, default=5)
* __--min_samples_regr__: Minimum number of **uncensored** samples in each **fold** for regression metric calculation (only used if aggregation_weight is not provided in --weights_regr) (int, default=10)
* __--dev__: Compute device to use (str, default="cuda:0", possible ["cpu","cuda:X"])
* __--run_name__: Run name for results (str)
* __--output_dir__: Output directory, including boards (str, default="models")
* __--prefix__: Prefix for run name (str, default='run')
* __--verbose__: Verbosity level: 2 = full; 1 = no progress; 0 = no output", type=int, default=2, choices=[0, 1, 2])
* __--save_model__: Set this to 0 if the model should not be saved (int, default=1)
* __--save_board__: Set this to 0 if the TensorBoard should not be saved
* __--eval_train__: Set this to 1 to calculate AUCs for train data (int, default=0)
* __--eval_frequency__: The gap between AUC eval (in epochs), -1 means to do an eval at the end. (int, default=1)
* __--optimizer__: Choose the optimizer (Adam or SGD, default: Adam)
* __--optimizer_params__: Set optimizer specific parameters (Adam(beta1, beta2, epsilon), SGD(momentum), default: Pytorch default)

