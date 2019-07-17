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
```

Then execute training:
```
cd ./examples/chembl
python chembl_train.py
```


