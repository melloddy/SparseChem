# Introduction

This package provide **fast** and **accurate** machine learning models for biochemical applications.
Especially, we support very high-dimensional models with sparse inputs, *e.g.*, millions of features and millions of compounds.
The general documentation can be found in this [folder](docs/).

# Local trunk branch

In this branch an extra script is created [retrain](examples/chembl/retrain.py). The objective for this script is to start with a given pretrained (federated) sparsechem model and integrate and fix the trunk of this model inside a new model. This new model will provide an extra local trunk and on top a new head. The local trunk and new head will be trained using local data while the give pretrained trunk will no be altered.

# How to use retrain script

The same parameters as the [train script](example/chembl/train.py) should be provided. These parameters are explained in detail [here](docs/main.md). Two extra parameters should be provided:

```
  --conf CONF           Model conf file (.json or .npy)
  --model MODEL         Pytorch model file (.pt)
```

The output of this script is similar as the regular [train script](example/chembl/train.py).
