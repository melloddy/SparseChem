# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import scipy.io
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--x", help="Descriptor file (matrix market or numpy)", type=str, default="chembl_23_x.mtx")
parser.add_argument("--y", help="Activity file (matrix market or numpy)", type=str, default="chembl_23_y.mtx")
parser.add_argument("--task_info", help="CSV file with columns task_id , weight and type (optional: default classification)", type=str, default=None)
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
parser.add_argument("--min_samples_auc", help="Minimum number samples for AUC calculation", type=int, default=25)
parser.add_argument("--dev", help="Device to use", type=str, default="cuda:0")
parser.add_argument("--filename", help="Filename for results", type=str, default=None)
parser.add_argument("--prefix", help="Prefix for run name (default 'run')", type=str, default='run')
parser.add_argument("--verbose", help="Verbosity level: 2 = full; 1 = no progress; 0 = no output", type=int, default=2, choices=[0, 1, 2])
parser.add_argument("--save_model", help="Set this to 0 if the model should not be saved", type=int, default=1)
parser.add_argument("--eval_train", help="Set this to 1 to calculate AUCs for train data", type=int, default=0)
parser.add_argument("--eval_frequency", help="The gap between AUC eval (in epochs), -1 means to do an eval at the end.", type=int, default=1)

args = parser.parse_args()

def vprint(s=""):
    if args.verbose:
        print(s)

vprint(args)

if args.filename is not None:
    name = args.filename
else:
    name  = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo{args.last_dropout:.1f}_wd{args.weight_decay}"
    name += f"_lr{args.lr}_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
    name += f"_fva{args.fold_va}_fte{args.fold_te}"
vprint(f"Run name is '{name}'.")

tb_name = "runs/"+name
writer = SummaryWriter(tb_name)
assert args.input_size_freq is None, "Using tail compression not yet supported."

ecfp = sc.load_sparse(args.x)
if ecfp is None:
   parser.print_help()
   vprint("--x: Descriptor file must have suffix .mtx or .npy")
   sys.exit(1)

ic50 = sc.load_sparse(args.y)
if ic50 is None:
   parser.print_help()
   vprint("--y: Activity file must have suffix .mtx or .npy")
   sys.exit(1)

folding = np.load(args.folding)

## Loading task weights
if args.task_info is not None:
    tw_df = pd.read_csv(args.task_info)
    assert "task_id" in tw_df.columns, "task_id is missing in task info CVS file"
    assert "weight" in tw_df.columns, "weight is missing in task info CSV file"
    
    tw_df.sort_values("task_id", inplace=True)
    
    if tw_df.shape[1] == 2:
        task_types = np.ones(ic50.shape[1], dtype=np.int16)
    else:
        assert tw_df.shape[1] == 3, "Task weight file (CSV) can only have 2 or max 3 columns"
        assert "task_type" in tw_df.columns, "task_type is missing in task info CSV file"
        assert (tw_df.task_type == 1 or tw_df.task_type == 2 or tw_df.task_ype == 3 or tw_df.task_type == 4).all(), "task type can only be 1,2,3 or 4"
        task_types = tw_df.task_type.values.astype(np.int16)

    assert ic50.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
    assert (0 <= tw_df.weight).all(), "task weights must not be negative"
    assert (tw_df.weight <= 1).all(), "task weights must not be larger than 1.0"

    assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
    assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
    assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
    assert tw_df.shape[0]==ic50.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({ic50.shape[1]})."

    task_weights = tw_df.weight.values.astype(np.float32)
else:
    ## default weights are set to 1.0
    task_weights = np.ones(ic50.shape[1], dtype=np.float32)
    task_types = np.ones(ic50.shape[1], dtype=np.int16)


assert ecfp.shape[0] == ic50.shape[0]
assert ecfp.shape[0] == folding.shape[0]

if args.fold_inputs is not None:
    ecfp = sc.fold_inputs(ecfp, folding_size=args.fold_inputs)
    vprint(f"Folding inputs to {ecfp.shape[1]} dimensions.")

## Input transformation
if args.input_transform == "binarize":
    ecfp.data = (ecfp.data > 0).astype(np.float)
elif args.input_transform == "tanh":
    ecfp.data = np.tanh(ecfp.data)
elif args.input_transform == "none":
    pass

num_pos  = np.array((ic50 == +1).sum(0)).flatten()
num_neg  = np.array((ic50 == -1).sum(0)).flatten()
auc_cols = np.where((num_pos >= args.min_samples_auc) & (num_neg >= args.min_samples_auc))[0]

vprint(f"There are {len(auc_cols)} columns for calculating mean AUC (i.e., have at least {args.min_samples_auc} positives and {args.min_samples_auc} negatives).")
vprint(f"Input dimension: {ecfp.shape[1]}")
vprint(f"#samples:        {ecfp.shape[0]}")
vprint(f"#tasks:          {ic50.shape[1]}")

if args.fold_te is not None:
    ## removing test data
    assert args.fold_te != args.fold_va, "fold_va and fold_te must not be equal."
    keep    = folding != args.fold_te
    ecfp    = ecfp[keep]
    ic50    = ic50[keep]
    folding = folding[keep]

fold_va = args.fold_va
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

num_pos_va  = np.array((ic50[idx_va] == +1).sum(0)).flatten() 
num_neg_va  = np.array((ic50[idx_va] == -1).sum(0)).flatten()

batch_size  = int(np.ceil(args.batch_ratio * idx_tr.shape[0]))
num_int_batches = 1

if args.internal_batch_max is not None:
    if args.internal_batch_max < batch_size:
        num_int_batches = int(np.ceil(batch_size / args.internal_batch_max))
        batch_size      = int(np.ceil(batch_size / num_int_batches))
vprint(f"#internal batch size:   {batch_size}")

dataset_tr = sc.SparseDataset(x=ecfp[idx_tr], y=ic50[idx_tr])
dataset_va = sc.SparseDataset(x=ecfp[idx_va], y=ic50[idx_va])

loader_tr  = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=sc.sparse_collate, shuffle=True)
loader_va  = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

args.input_size  = dataset_tr.input_size
args.output_size = dataset_tr.output_size

dev  = args.dev
net  = sc.SparseFFN(args).to(dev)
loss = torch.nn.BCEWithLogitsLoss(reduction="none")

vprint("Network:")
vprint(net)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_alpha)

task_weights = torch.from_numpy(task_weights).to(dev)

num_prints = 0

for epoch in range(args.epochs):
    t0 = time.time()
    loss_tr = sc.train_binary(
        net, optimizer, loader_tr, loss, dev,
        task_weights    = task_weights,
        num_int_batches = num_int_batches,
        progress        = args.verbose >= 2)

    t1 = time.time()

    eval_round = (args.eval_frequency > 0) and ((epoch + 1) % args.eval_frequency == 0)
    last_round = epoch == args.epochs - 1

    if eval_round or last_round:
        results_va = sc.evaluate_binary(net, loader_va, loss, dev, progress = args.verbose >= 2)
        t2 = time.time()

        if args.eval_train:
            results_tr = sc.evaluate_binary(net, loader_tr, loss, dev, progress = args.verbose >= 2)
            metrics_tr = results_tr["metrics"].reindex(labels=auc_cols).mean(0)
            metrics_tr["epoch_time"] = t1 - t0
            metrics_tr["logloss"]    = results_tr['logloss']
            for metric_tr_name in metrics_tr.index:
                writer.add_scalar(metric_tr_name+"/tr", metrics_tr[metric_tr_name], epoch)
        else:
            results_tr = None
            metrics_tr = None

        metrics_va = results_va["metrics"].reindex(labels=auc_cols).mean(0)
        metrics_va["epoch_time"] = t2 - t1
        metrics_va["logloss"]    = results_va["logloss"]
        for metric_va_name in metrics_va.index:
            writer.add_scalar(metric_va_name+"/va", metrics_va[metric_va_name], epoch)

        if args.verbose:
            header = num_prints % 20 == 0
            num_prints += 1
            sc.print_metrics(epoch, t1 - t0, metrics_tr, metrics_va, header)

    scheduler.step()

writer.close()
vprint()
vprint("Saving performance metrics (AUCs) and model.")

#####   model saving   #####
if not os.path.exists("models"):
   os.makedirs("models")

model_file = f"models/{name}.pt"
out_file   = f"models/{name}.json"

if args.save_model:
   torch.save(net.state_dict(), model_file)
   vprint(f"Saved model weights into '{model_file}'.")

## adding positive and negative numbers
results_va["metrics"]["num_pos"] = num_pos_va
results_va["metrics"]["num_neg"] = num_neg_va

out = {}
out["conf"]        = args.__dict__
out["results"]     = {"va": results_va["metrics"].to_json()}
out["results_agg"] = {"va": metrics_va.to_json()}

if args.eval_train:
    results_tr["metrics"]["num_pos"] = num_pos - num_pos_va
    results_tr["metrics"]["num_neg"] = num_neg - num_neg_va

    out["results"]["tr"]     = results_tr["metrics"].to_json()
    out["results_agg"]["tr"] = metrics_tr.to_json()

with open(out_file, "w") as f:
    json.dump(out, f)

vprint(f"Saved config and results into '{out_file}'.\nYou can load the results by:\n  import sparsechem as sc\n  res = sc.load_results('{out_file}')")
