# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import tqdm
import argparse
import os
import sys
import os.path
import time
import json
from sparsechem import Nothing
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--x", help="Descriptor file (matrix market or numpy)", type=str, default=None)
parser.add_argument("--y_class", "--y", "--y_classification", help="Activity file (matrix market or numpy)", type=str, default=None)
parser.add_argument("--y_regr", "--y_regression", help="Activity file (matrix market or numpy)", type=str, default=None)
parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, weight (for classification tasks)", type=str, default=None)
parser.add_argument("--weights_regr", "--weights_regression", help="CSV file with columns task_id, weight (for regression tasks)", type=str, default=None)
parser.add_argument("--folding", help="Folding file (npy)", type=str, default="folding_hier_0.6.npy")
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
parser.add_argument("--fold_te", help="Test fold number (removed from dataset)", type=int, default=None)
parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.02)
parser.add_argument("--internal_batch_max", help="Maximum size of the internal batch", type=int, default=None)
parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes", default=[], type=int, required=True)
parser.add_argument("--middle_dropout", help="Dropout for layers before the last", type=float, default=0.0)
parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
parser.add_argument("--last_non_linearity", help="Last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
parser.add_argument("--non_linearity", help="Before last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="binarize", choices=["binarize", "none", "tanh"])
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])
parser.add_argument("--input_size_freq", help="Number of high importance features", type=int, default=None)
parser.add_argument("--fold_inputs", help="Fold input to a fixed set (default no folding)", type=int, default=None)
parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
parser.add_argument("--min_samples_auc", help="Minimum number samples (in each class) for AUC calculation", type=int, default=25)
parser.add_argument("--min_samples_regr", help="Minimum number samples for regression metric calculation", type=int, default=100)
parser.add_argument("--dev", help="Device to use", type=str, default="cuda:0")
parser.add_argument("--run_name", help="Run name for results", type=str, default=None)
parser.add_argument("--output_dir", help="Output directory, including boards (default 'models')", type=str, default="models")
parser.add_argument("--prefix", help="Prefix for run name (default 'run')", type=str, default='run')
parser.add_argument("--verbose", help="Verbosity level: 2 = full; 1 = no progress; 0 = no output", type=int, default=2, choices=[0, 1, 2])
parser.add_argument("--save_model", help="Set this to 0 if the model should not be saved", type=int, default=1)
parser.add_argument("--save_board", help="Set this to 0 if the TensorBoard should not be saved", type=int, default=1)
parser.add_argument("--eval_train", help="Set this to 1 to calculate AUCs for train data", type=int, default=0)
parser.add_argument("--eval_frequency", help="The gap between AUC eval (in epochs), -1 means to do an eval at the end.", type=int, default=1)

args = parser.parse_args()

def vprint(s=""):
    if args.verbose:
        print(s)

vprint(args)

if args.run_name is not None:
    name = args.run_name
else:
    name  = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo{args.last_dropout:.1f}_wd{args.weight_decay}"
    name += f"_lr{args.lr}_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
    name += f"_fva{args.fold_va}_fte{args.fold_te}"
vprint(f"Run name is '{name}'.")

if args.save_board:
    tb_name = os.path.join(args.output_dir, "boards", name)
    writer  = SummaryWriter(tb_name)
else:
    writer = Nothing()
assert args.input_size_freq is None, "Using tail compression not yet supported."

if (args.y_class is None) and (args.y_regr is None):
    raise ValueError("No label data specified, please add --y_class and/or --y_regr.")

ecfp    = sc.load_sparse(args.x)
y_class = sc.load_sparse(args.y_class)
y_regr  = sc.load_sparse(args.y_regr)

if y_class is None:
    y_class = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
if y_regr is None:
    y_regr  = scipy.sparse.csr_matrix((ecfp.shape[0], 0))

folding = np.load(args.folding)
assert ecfp.shape[0] == folding.shape[0], "x and folding must have same number of rows"

## Loading task weights
weights_class = sc.load_task_weights(args.weights_class, y=y_class, label="y_class")
weights_regr  = sc.load_task_weights(args.weights_regr, y=y_regr, label="y_regr")

if args.fold_inputs is not None:
    ecfp = sc.fold_inputs(ecfp, folding_size=args.fold_inputs)
    vprint(f"Folding inputs to {ecfp.shape[1]} dimensions.")

## Input transformation
ecfp = sc.fold_transform_inputs(ecfp, folding_size=args.fold_inputs, transform=args.input_transform)

num_pos    = np.array((y_class == +1).sum(0)).flatten()
num_neg    = np.array((y_class == -1).sum(0)).flatten()
num_regr   = np.bincount(y_regr.indices)
class_cols = np.where((num_pos >= args.min_samples_auc) & (num_neg >= args.min_samples_auc))[0]
regr_cols  = np.where(num_regr >= args.min_samples_regr)[0]

vprint(f"Input dimension: {ecfp.shape[1]}")
vprint(f"#samples:        {ecfp.shape[0]}")
vprint(f"#classification tasks:  {y_class.shape[1]}")
vprint(f"#regression tasks:      {y_regr.shape[1]}")
vprint(f"There are {len(class_cols)} classification tasks for calculating mean AUC (i.e., have at least {args.min_samples_auc} positives and {args.min_samples_auc} negatives).")
vprint(f"There are {len(regr_cols)} regression tasks for calculating metrics (i.e., having at least {args.min_samples_regr} data points).")

if args.fold_te is not None:
    ## removing test data
    assert args.fold_te != args.fold_va, "fold_va and fold_te must not be equal."
    keep    = folding != args.fold_te
    ecfp    = ecfp[keep]
    y_class = y_class[keep]
    y_regr  = y_regr[keep]
    folding = folding[keep]

fold_va = args.fold_va
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

y_class_tr = y_class[idx_tr]
y_class_va = y_class[idx_va]
y_regr_tr  = y_regr[idx_tr]
y_regr_va  = y_regr[idx_va]

num_pos_va  = np.array((y_class_va == +1).sum(0)).flatten()
num_neg_va  = np.array((y_class_va == -1).sum(0)).flatten()
num_regr_va = np.bincount(y_regr_va.indices)

batch_size  = int(np.ceil(args.batch_ratio * idx_tr.shape[0]))
num_int_batches = 1

if args.internal_batch_max is not None:
    if args.internal_batch_max < batch_size:
        num_int_batches = int(np.ceil(batch_size / args.internal_batch_max))
        batch_size      = int(np.ceil(batch_size / num_int_batches))
vprint(f"#internal batch size:   {batch_size}")

dataset_tr = sc.ClassRegrSparseDataset(x=ecfp[idx_tr], y_class=y_class_tr, y_regr=y_regr_tr)
dataset_va = sc.ClassRegrSparseDataset(x=ecfp[idx_va], y_class=y_class_va, y_regr=y_regr_va)

loader_tr = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=dataset_tr.collate, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=dataset_va.collate, shuffle=False)

args.input_size  = dataset_tr.input_size
args.output_size = dataset_tr.output_size

args.class_output_size = dataset_tr.class_output_size
args.regr_output_size  = dataset_tr.regr_output_size

dev  = torch.device(args.dev)
net  = sc.SparseFFN(args).to(dev)
loss_class = torch.nn.BCEWithLogitsLoss(reduction="none")
loss_regr  = sc.censored_mse_loss

if weights_class is not None: weights_class = weights_class.to(dev)
if weights_regr is not None: weights_regr = weights_regr.to(dev)

vprint("Network:")
vprint(net)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_alpha)

num_prints = 0

for epoch in range(args.epochs):
    t0 = time.time()
    sc.train_class_regr(
        net, optimizer,
        loader          = loader_tr,
        loss_class      = loss_class,
        loss_regr       = loss_regr,
        dev             = dev,
        weights_class   = weights_class,
        weights_regr    = weights_regr,
        num_int_batches = num_int_batches,
        progress        = args.verbose >= 2)

    t1 = time.time()

    eval_round = (args.eval_frequency > 0) and ((epoch + 1) % args.eval_frequency == 0)
    last_round = epoch == args.epochs - 1

    if eval_round or last_round:
        results_va = sc.evaluate_class_regr(net, loader_va, loss_class, loss_regr, weights_class=weights_class, weights_regr=weights_regr, class_cols=class_cols, regr_cols=regr_cols, dev=dev, progress = args.verbose >= 2)
        for key, val in results_va["classification_agg"].items():
            writer.add_scalar(key+"/tr", val, epoch)
        for key, val in results_va["regression_agg"].items():
            writer.add_scalar(key+"/tr", val, epoch)

        if args.eval_train:
            results_tr = sc.evaluate_class_regr(net, loader_tr, loss_class, loss_regr, weights_class=weights_class, weights_regr=weights_regr, class_cols=class_cols, regr_cols=regr_cols, dev=dev, progress = args.verbose >= 2)
            for key, val in results_tr["classification_agg"].items():
                writer.add_scalar(key+"/tr", val, epoch)
            for key, val in results_tr["regression_agg"].items():
                writer.add_scalar(key+"/tr", val, epoch)
        else:
            results_tr = None

        if args.verbose:
            ## printing a new header every 20 lines
            header = num_prints % 20 == 0
            num_prints += 1
            sc.print_metrics_cr(epoch, t1 - t0, results_tr, results_va, header)

    scheduler.step()

writer.close()
vprint()
vprint("Saving performance metrics (AUCs) and model.")

#####   model saving   #####
if not os.path.exists(args.output_dir):
   os.makedirs(args.output_dir)

model_file = f"{args.output_dir}/{name}.pt"
out_file   = f"{args.output_dir}/{name}.json"

if args.save_model:
   torch.save(net.state_dict(), model_file)
   vprint(f"Saved model weights into '{model_file}'.")

results_va["classification"]["num_pos"] = num_pos_va
results_va["classification"]["num_neg"] = num_neg_va
results_va["regression"]["num_samples"] = num_regr_va

if results_tr is not None:
    results_tr["classification"]["num_pos"] = num_pos - num_pos_va
    results_tr["classification"]["num_neg"] = num_neg - num_neg_va
    results_tr["regression"]["num_samples"] = num_regr - num_regr_va

sc.save_results(out_file, args, validation=results_va, training=results_tr)

vprint(f"Saved config and results into '{out_file}'.\nYou can load the results by:\n  import sparsechem as sc\n  res = sc.load_results('{out_file}')")
