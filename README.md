# Introduction

This package provide **fast** and **accurate** machine learning models for biochemical applications.
Especially, we support very high-dimensional models with sparse inputs, *e.g.*, millions of features and millions of compounds.
The general documentation can be found in this [folder](docs/).

# Local trunk branch

In this branch an extra script is created: [retrain.py](examples/chembl/retrain.py). The objective for this script is to start with a given pretrained (federated) sparsechem model and integrate and fix the trunk of this model inside a new model. This new model will provide an extra local trunk and on top a new head. The local trunk and new head will be trained using local data while the given pretrained trunk will not be altered.

# How to use retrain script

The same parameters as the [train script](example/chembl/train.py) should be provided. These parameters are explained in detail [here](docs/main.md). Two extra parameters should be provided:

```
  --conf CONF           Model conf file (.json or .npy)
  --model MODEL         Pytorch model file (.pt)
```

The output of this script is similar as the regular [train script](example/chembl/train.py).

# Fold setup

In the federated run these folds were used:

```
FOLD_TEST=0
FOLD_VALIDATION=1
```

after hyperparameter tuning, the federated model was trained in phase 2 also including the `FOLD_VALIDATION=1` only leaving out `FOLD_TEST=0`. So for local retraining using local trunk the suggestion is to split up this `FOLD_TEST=0 `in `FOLD_LOCAL_VAL=0.1` and `FOLD_LOCAL_TEST=0.2`. However the way how to do it is not defined yet and still under discussion. So for the first 'sanity tests' the only fold available to evaluate the trained model (including HP tuning which is a problem for overfitting) is `FOLD=0`.
