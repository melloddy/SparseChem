# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import sys
import argparse
from torch.utils.data import DataLoader
from scipy.special import expit
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--x", help="Descriptor file (matrix market or numpy)", type=str, required=True)
parser.add_argument("--outprefix", help="Prefix for output files, '-class.npy', '-regr.npy' will be appended.", type=str, required=True)
parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--model", help="Pytorch model file (.pt)", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size (default 4000)", type=int, default=4000)
parser.add_argument("--last_hidden", help="If set to 1 returns last hidden layer instead of Yhat", type=int, default=0)
parser.add_argument("--dropout", help="If set to 1 enables dropout for evaluation", type=int, default=0)
parser.add_argument("--dev", help="Device to use (default cuda:0)", type=str, default="cuda:0")

args = parser.parse_args()

print(args)

conf = sc.load_results(args.conf, two_heads=True)["conf"]

ecfp = sc.load_sparse(args.x)
ecfp = sc.fold_transform_inputs(ecfp, folding_size=conf.fold_inputs, transform=conf.input_transform)

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

y0         = scipy.sparse.coo_matrix((ecfp.shape[0], conf.output_size), np.float32).tocsr()
dataset_te = sc.SparseDataset(x=ecfp, y=y0)
loader_te  = DataLoader(dataset_te, batch_size=args.batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

out = sc.predict(net, loader_te, dev, last_hidden=args.last_hidden, dropout=args.dropout)
class_out, regr_out = out

if class_out is not None:
    if args.last_hidden == 0:
        class_out = torch.sigmoid(class_out)
    class_out = class_out.numpy()
    np.save(f"{args.outprefix}-class.npy", class_out)
    print(f"Saved prediction matrix (numpy) for classification to '{args.outprefix}-class.npy'.")

if regr_out is not None:
    regr_out = regr_out.numpy()
    np.save(f"{args.outprefix}-regr.npy", regr_out)
    print(f"Saved prediction matrix (numpy) for regression to '{args.outprefix}-regr.npy'.")


