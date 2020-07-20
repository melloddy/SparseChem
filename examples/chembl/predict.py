# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
import numpy as np
import pandas as pd
import torch
import tqdm
import sys
import argparse
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from scipy.special import expit
from collections import OrderedDict

def keep_rows(y, keep):
    """
    Filters out data where keep is False. Output shape is same as 'y.shape'.
    Args:
        y     sparse matrix
        keep  bool vector, which rows' data to keep. If keep[i] is False i-th row data is removed
    """
    ycoo = y.tocoo()
    mask = keep[ycoo.row]
    return csr_matrix((ycoo.data[mask], (ycoo.row[mask], ycoo.col[mask])), shape=y.shape)

parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--x", help="Descriptor file (matrix market or numpy)", type=str, required=True)
parser.add_argument("--y", help="Activity file, optional. If provided returns predictions for given activities. (matrix market or numpy)", type=str, required=False)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--outfile", help="Output file for predictions (.npy)", type=str, required=True)
parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--model", help="Pytorch model file (.pt)", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size (default 4000)", type=int, default=4000)
parser.add_argument("--last_hidden", help="If set to 1 returns last hidden layer instead of Yhat", type=int, default=0)
parser.add_argument("--dropout", help="If set to 1 enables dropout for evaluation", type=int, default=0)
parser.add_argument("--dev", help="Device to use (default cuda:0)", type=str, default="cuda:0")

args = parser.parse_args()

print(args)

conf = sc.load_results(args.conf)["conf"]
ecfp = sc.load_sparse(args.x)
if ecfp is None:
   parser.print_help()
   print("--x: Descriptor file must have suffix .mtx or .npy")
   sys.exit(1) 

if conf.fold_inputs is not None:
    ecfp = sc.fold_inputs(ecfp, folding_size=conf.fold_inputs)
    print(f"Folding inputs to {ecfp.shape[1]} dimensions.")

## error checks for --y, --folding and --predict_fold
if args.last_hidden:
    assert args.y is None, "Cannot use '--last_hidden 1' with sparse predictions ('--y' is specified)."
if args.y is None:
    assert args.predict_fold is None, "To use '--predict_fold' please specify '--y'."
    assert args.folding is None, "To use '--folding' please specify '--y'."
else:
    if args.predict_fold is None:
        assert args.folding is None, "If --folding is given please also specify --predict_fold."
    if args.folding is None:
        assert args.predict_fold is None, "If --predict_fold is given please also specify --folding."

print(f"Input dimension: {ecfp.shape[1]}")
print(f"#samples:        {ecfp.shape[0]}")

dev = args.dev
net = sc.SparseFFN(conf).to(dev)
state_dict = torch.load(args.model, map_location=torch.device(dev))

if conf.model_type == "federated":
    state_dict_new = OrderedDict()
    state_dict_new["net.0.net_freq.weight"] = state_dict["0.0.net_freq.weight"]
    state_dict_new["net.0.net_freq.bias"]   = state_dict["0.0.net_freq.bias"]
    state_dict_new["net.2.net.2.weight"]    = state_dict["1.net.2.weight"]
    state_dict_new["net.2.net.2.bias"]      = state_dict["1.net.2.bias"]
    state_dict = state_dict_new

net.load_state_dict(state_dict)
print(f"Model weights:   '{args.model}'")
print(f"Model config:    '{args.conf}'.")

## creating/loading y
if args.y is None:
    y = csr_matrix((ecfp.shape[0], conf.output_size), dtype=np.float32)
else:
    y = sc.load_sparse(args.y)
    assert y is not None, f"Unsupported filetype for --y: '{args.y}'."
    assert ecfp.shape[0] == y.shape[0], f"The number of rows in X ({ecfp.shape[0]}) must be equal to the number of rows in Y ({y.shape[0]})."
    assert y.shape[1] == conf.output_size, f"Y matrix has {y.shape[1]} columns and model has {conf.output_size}. They must be equal."
    if args.predict_fold is not None:
        folding = np.load(args.folding)
        assert folding.shape[0] == ecfp.shape[0], f"Folding has {folding.shape[0]} rows and X has {ecfp.shape[0]}. Must be equal."
        keep    = np.isin(folding, args.predict_fold)
        y       = keep_rows(y, keep)

dataset_te = sc.SparseDataset(x=ecfp, y=y)
loader_te  = DataLoader(dataset_te, batch_size=args.batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

if args.y is None:
    out = sc.predict(net, loader_te, dev, last_hidden=args.last_hidden, dropout=args.dropout)
    out = out.numpy()
else:
    out = sc.predict_sparse(net, loader_te, dev, dropout=args.dropout, progress=True)

np.save(args.outfile, out)
print(f"Saved prediction matrix (numpy) to '{args.outfile}'.")

