# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
from torch import nn
import numpy as np
import pandas as pd
import torch
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

def unstack_SparseFFN_model(model):
    sparse_input = model.net[0]
    middle_net   = model.net[1]
    head         = model.net[2]
    trunk = nn.Sequential(
            sparse_input,
            middle_net,
    )
    return head, trunk

parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, required=True)
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_regr", "--y_regression", help="Sparse pattern file for regression, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--outprefix", help="Prefix for output files, '-class.npy', '-regr.npy' will be appended.", type=str, required=True)
parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--fedconf", help="Federated Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--model", help="Pytorch model file (.pt)", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size (default 4000)", type=int, default=4000)
parser.add_argument("--last_hidden", help="If set to 1 returns last hidden layer instead of Yhat", type=int, default=0)
parser.add_argument("--dropout", help="If set to 1 enables dropout for evaluation", type=int, default=0)
parser.add_argument("--disable_localtrunk", help="If set to 1 disables dropout localtrunk", type=int, default=0)
parser.add_argument("--dev", help="Device to use (default cuda:0)", type=str, default="cuda:0")

args = parser.parse_args()

print(args)

conf = sc.load_results(args.conf, two_heads=True)["conf"]
fedconf = sc.load_results(args.fedconf, two_heads=True)["conf"]
if fedconf.last_hidden_sizes is None:
   setattr(fedconf, "last_hidden_sizes", [])
x = sc.load_sparse(args.x)
x = sc.fold_transform_inputs(x, folding_size=conf.fold_inputs, transform=conf.input_transform)

print(f"Input dimension: {x.shape[1]}")
print(f"#samples:        {x.shape[0]}")

## error checks for --y_class, --y_regr, --folding and --predict_fold
if args.last_hidden:
    assert args.y_class is None, "Cannot use '--last_hidden 1' with sparse predictions ('--y_class' or '--y_regr' is specified)."


if args.y_class is None and args.y_regr is None:
    assert args.predict_fold is None, "To use '--predict_fold' please specify '--y_class' and/or '--y_regr'."
    assert args.folding is None, "To use '--folding' please specify '--y_class' and/or '--y_regr'."
else:
    if args.predict_fold is None:
        assert args.folding is None, "If --folding is given please also specify --predict_fold."
    if args.folding is None:
        assert args.predict_fold is None, "If --predict_fold is given please also specify --folding."

dev  = torch.device(args.dev)
#import ipdb; ipdb.set_trace()
fednet = sc.SparseFFN(fedconf).to(dev)
fed_head, fed_trunk = unstack_SparseFFN_model(fednet)
if args.disable_localtrunk == 0:
   newhead    = sc.LastNet(conf, fedconf.hidden_sizes[-1])
   local_trunk = nn.Sequential(
                sc.SparseInputNet(conf),
                sc.MiddleNet(conf)
             )
else:
    conf.hidden_sizes = fedconf.hidden_sizes
    newhead = sc.LastNet(conf)
    local_trunk = None
net = sc.SparseFFN_combined(conf, fed_trunk, local_trunk, newhead).to(dev)
#import ipdb; ipdb.set_trace()
state_dict = torch.load(args.model, map_location=torch.device(dev))


net.load_state_dict(state_dict)
print(f"Model weights:   '{args.model}'")
print(f"Model config:    '{args.conf}'.")
print(f"FedModel config:    '{args.fedconf}'.")

y_class = sc.load_check_sparse(args.y_class, (x.shape[0], conf.class_output_size))
y_regr  = sc.load_check_sparse(args.y_regr, (x.shape[0], conf.regr_output_size))

if args.folding is not None:
    folding = np.load(args.folding) if args.folding else None
    assert folding.shape[0] == x.shape[0], f"Folding has {folding.shape[0]} rows and X has {x.shape[0]}. Must be equal."
    keep    = np.isin(folding, args.predict_fold)
    y_class = sc.keep_row_data(y_class, keep)
    y_regr  = sc.keep_row_data(y_regr, keep)

dataset_te = sc.ClassRegrSparseDataset(x=x, y_class=y_class, y_regr=y_regr)
loader_te  = DataLoader(dataset_te, batch_size=args.batch_size, num_workers = 4, pin_memory=True, collate_fn=dataset_te.collate)

if args.last_hidden:
    ## saving only hidden layer
    out      = sc.predict_hidden(net, loader_te, dev=dev, dropout=args.dropout, progress=True)
    filename = f"{args.outprefix}-hidden.npy"
    np.save(filename, out.numpy())
    print(f"Saved (numpy) matrix of hiddens to '{filename}'.")
else:
    if args.y_class is None and args.y_regr is None:
        class_out, regr_out = sc.predict_dense(net, loader_te, dev=dev, dropout=args.dropout, progress=True)
    else:
        class_out, regr_out = sc.predict_sparse(net, loader_te, dev=dev, dropout=args.dropout, progress=True)

    if net.class_output_size > 0:
        np.save(f"{args.outprefix}-class.npy", class_out)
        print(f"Saved prediction matrix (numpy) for classification to '{args.outprefix}-class.npy'.")
    if net.regr_output_size > 0:
        np.save(f"{args.outprefix}-regr.npy", regr_out)
        print(f"Saved prediction matrix (numpy) for regression to '{args.outprefix}-regr.npy'.")

