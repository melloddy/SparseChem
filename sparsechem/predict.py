# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
import numpy as np
import types
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

def predict():
    parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
    parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, required=True)
    parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--y_regr", "--y_regression", help="Sparse pattern file for regression, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
    parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
    parser.add_argument("--outprefix", help="Prefix for output files, '-class.npy', '-regr.npy' will be appended.", type=str, required=True)
    parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
    parser.add_argument("--model", help="Pytorch model file (.pt)", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size (default 4000)", type=int, default=4000)
#    parser.add_argument("--last_hidden", help="If set to 1 returns last hidden layer instead of Yhat", type=int, default=0)
    parser.add_argument("--trunk_embeddings", help="If set to 1 return trunk embeddings (before non-linearity, e.g relu,tanh)  instead of Yhat", type=int, default=0)
    parser.add_argument("--dropout", help="If set to 1 enables dropout for evaluation", type=int, default=0)
    parser.add_argument("--inverse_normalization", help="If set to 1 enables inverse normalization given means and variances from config file", type=int, default=0)
    parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks)", type=str, default=None)
    parser.add_argument("--dev", help="Device to use (default cuda:0)", type=str, default="cuda:0")
    parser.add_argument("--num_workers", help="Number of workers for DataLoader", type=int, default=4)

    args = parser.parse_args()

    print(args)

    results_loaded = sc.load_results(args.conf, two_heads=True)
    conf  = results_loaded["conf"]
    if args.inverse_normalization == 1:
        stats = results_loaded["stats"]

    x = sc.load_sparse(args.x)
    x = sc.fold_transform_inputs(x, folding_size=conf.fold_inputs, transform=conf.input_transform)

    print(f"Input dimension: {x.shape[1]}")
    print(f"#samples:        {x.shape[0]}")

    ## error checks for --y_class, --y_regr, --folding and --predict_fold
#    if args.last_hidden:
#        assert args.y_class is None, "Cannot use '--last_hidden 1' with sparse predictions ('--y_class' or '--y_regr' is specified)."


    if args.y_class is None and args.y_regr is None:
        assert args.predict_fold is None, "To use '--predict_fold' please specify '--y_class' and/or '--y_regr'."
        assert args.folding is None, "To use '--folding' please specify '--y_class' and/or '--y_regr'."
    else:
        if args.predict_fold is None:
            assert args.folding is None, "If --folding is given please also specify --predict_fold."
        if args.folding is None:
            assert args.predict_fold is None, "If --predict_fold is given please also specify --folding."

    res = types.SimpleNamespace(task_id=None, training_weight=None, aggregation_weight=None, task_type=None, censored_weight=torch.FloatTensor(), cat_id=None)
    if args.weights_class is not None:
       tasks_class = pd.read_csv(args.weights_class)
       if "catalog_id" in tasks_class:
            res.cat_id = tasks_class.catalog_id.values
    tasks_cat_id_list = None
    select_cat_ids = None
    if res.cat_id is not None:
        tasks_cat_id_list = [[x,i] for i,x in enumerate(res.cat_id) if str(x) != 'nan']
        tasks_cat_ids = [i for i,x in enumerate(res.cat_id) if str(x) != 'nan']
        select_cat_ids = np.array(tasks_cat_ids)
        cat_id_size = len(tasks_cat_id_list)
    else:
        cat_id_size = 0

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

    y_class = sc.load_check_sparse(args.y_class, (x.shape[0], conf.class_output_size))
    y_regr  = sc.load_check_sparse(args.y_regr, (x.shape[0], conf.regr_output_size))

    if args.folding is not None:
        folding = np.load(args.folding) if args.folding else None
        assert folding.shape[0] == x.shape[0], f"Folding has {folding.shape[0]} rows and X has {x.shape[0]}. Must be equal."
        keep    = np.isin(folding, args.predict_fold)
        y_class = sc.keep_row_data(y_class, keep)
        y_regr  = sc.keep_row_data(y_regr, keep)

    dataset_te = sc.ClassRegrSparseDataset(x=x, y_class=y_class, y_regr=y_regr)
    loader_te  = DataLoader(dataset_te, batch_size=args.batch_size, num_workers = args.num_workers, pin_memory=True, collate_fn=dataset_te.collate)

    if args.trunk_embeddings:
        ## saving only hidden layer
        out      = sc.predict_hidden(net, loader_te, dev=dev, dropout=args.dropout, progress=True)
        filename = f"{args.outprefix}-hidden.npy"
        np.save(filename, out.numpy())
        print(f"Saved (numpy) matrix of hiddens to '{filename}'.")
    else:
        if args.y_class is None and args.y_regr is None:
            class_out, regr_out = sc.predict_dense(net, loader_te, dev=dev, dropout=args.dropout, progress=True, y_cat_columns=select_cat_ids)
        else:
            class_out, regr_out = sc.predict_sparse(net, loader_te, dev=dev, dropout=args.dropout, progress=True, y_cat_columns=select_cat_ids)
            if args.inverse_normalization == 1:
               regr_out = sc.inverse_normalization(regr_out, mean=np.array(stats["mean"]), variance=np.array(stats["var"]), array=True)
        if net.class_output_size > 0:
            np.save(f"{args.outprefix}-class.npy", class_out)
            print(f"Saved prediction matrix (numpy) for classification to '{args.outprefix}-class.npy'.")
        if net.regr_output_size > 0:
            np.save(f"{args.outprefix}-regr.npy", regr_out)
            print(f"Saved prediction matrix (numpy) for regression to '{args.outprefix}-regr.npy'.")

if __name__ == "__main__":
    predict()
