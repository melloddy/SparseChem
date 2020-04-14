# Introduction

This package provide **fast** and **accurate** machine learning models for biochemical applications.
Especially, we support very high-dimensional models with sparse inputs, *e.g.*, millions of features and millions of compounds.

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
  --y ./chembl_23_y.mtx \
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
We also modified the weight decay to `1e-5`.

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
Sparsechem also supports task weighting.
This can be enabled by adding a `--task_weights weights.csv` option,
where the file `weights.csv` should have two columns:
* `task_id` integer from 0 to Ntasks - 1,
* `weight` real value between 0.0 and 1.0 (inclusive).

The number of weights in the CSV file must be equal to the number of tasks in `y` matrix.

## Running on CPU or other GPUs
The default device is `cuda:0`.
To train the model on CPU just add `--dev cpu` to the arguments.
Similarly, to choose another GPU, we can specify `--dev cuda:1`.

## Predicting on new compounds
After the run is complete the model's **weights** and **conf** are saved under `models/` folder.
We then can use `predict.py` to make predictions for new compounds as follows:
```bash
python predict.py \
    --x new_compounds.mtx \
    --outfile y_hat.npy \
    --conf models/sc_chembl_h400.400_ldo0.2_wd1e-05.json \
    --model models/sc_chembl_h400.400_ldo0.2_wd1e-05.pt \
    --dev cuda:0
```
where `new_compounds.mtx` is the sparse feature matrix of the new compounds and `--outfile y_hat.npy` specifies the file where the predictions are saved to.
The `--conf` and `--model` should point to the configuration and model files that where saved during the training.

The format for the prediction is a Numpy file that can be loaded as follows:
```python
import numpy as np
y_hat = np.load("y_hat.npy")
```
The predictions themselves are class probabilities (values between 0.0 and 1.0).

There is an option `--dropout 1` to switch on the dropout during predictions to obtain stochastic predictions, *e.g.*, for MC-dropout. 

## Retreiving last hidden layers
Instead of outputting the predictions we can use `predict.py` to output the activations of the last layer.
This can be done by adding the option `--last_hidden 1`.

Then the output file will contain the numpy matrix of the hidden vectors, which can be loaded the same way as predictions.

