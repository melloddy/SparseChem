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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--x", help="Descriptor file (matrix market or numpy)", type=str, default="chembl_23_x.mtx")
parser.add_argument("--y", help="Activity file (matrix market or numpy)", type=str, default="chembl_23_y.mtx")
parser.add_argument("--task_weights", help="CSV file with columns task_id and weight", type=str, default=None)
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
parser.add_argument("--save_model", help="Set this to 0 if the model should not be saved", type=int, default=1)

args = parser.parse_args()

print(args)
if args.filename is not None:
    name = args.filename
else:
    name  = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo{args.last_dropout:.1f}_wd{args.weight_decay}"
    name += f"_lr{args.lr}_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
    name += f"_fva{args.fold_va}_fte{args.fold_te}"
print(f"Run name is '{name}'.")

tb_name = "runs/"+name
writer = SummaryWriter(tb_name)
assert args.input_size_freq is None, "Using tail compression not yet supported."

ecfp = sc.load_sparse(args.x)
if ecfp is None:
   parser.print_help()
   print("--x: Descriptor file must have suffix .mtx or .npy")
   sys.exit(1)

ic50 = sc.load_sparse(args.y)
if ic50 is None:
   parser.print_help()
   print("--y: Activity file must have suffix .mtx or .npy")
   sys.exit(1)

folding = np.load(args.folding)

## Loading task weights
if args.task_weights is not None:
    tw_df = pd.read_csv(args.task_weights)
    assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
    assert "weight" in tw_df.columns, "weight is missing in task weights CVS file"
    assert tw_df.shape[1] == 2, "Task weight file (CSV) must only have 2 columns"

    assert ic50.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
    assert (0 <= tw_df.weight).all(), "task weights must not be negative"
    assert (tw_df.weight <= 1).all(), "task weights must not be larger than 1.0"

    assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
    assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
    assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
    assert tw_df.shape[0]==ic50.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({ic50.shape[1]})."

    tw_df.sort_values("task_id", inplace=True)
    task_weights = tw_df.weight.values.astype(np.float32)
else:
    ## default weights are set to 1.0
    task_weights = np.ones(ic50.shape[1], dtype=np.float32)


assert ecfp.shape[0] == ic50.shape[0]
assert ecfp.shape[0] == folding.shape[0]

if args.fold_inputs is not None:
    ecfp = sc.fold_inputs(ecfp, folding_size=args.fold_inputs)
    print(f"Folding inputs to {ecfp.shape[1]} dimensions.")

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

print(f"There are {len(auc_cols)} columns for calculating mean AUC (i.e., have at least {args.min_samples_auc} positives and {args.min_samples_auc} negatives).")
print(f"Input dimension: {ecfp.shape[1]}")
print(f"#samples:        {ecfp.shape[0]}")
print(f"#tasks:          {ic50.shape[1]}")

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
print(f"#internal batch size:   {batch_size}")

dataset_tr = sc.SparseDataset(x=ecfp[idx_tr], y=ic50[idx_tr])
dataset_va = sc.SparseDataset(x=ecfp[idx_va], y=ic50[idx_va])

loader_tr  = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=sc.sparse_collate, shuffle=True)
loader_va  = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

args.input_size  = dataset_tr.input_size
args.output_size = dataset_tr.output_size

dev  = args.dev
net  = sc.SparseFFN(args).to(dev)
loss = torch.nn.BCEWithLogitsLoss(reduction="none")

print("Network:")
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_alpha)

task_weights = torch.from_numpy(task_weights).to(dev)

for epoch in range(args.epochs):
    t0 = time.time()
    loss_tr = sc.train_binary(
            net, optimizer, loader_tr, loss, dev,
            task_weights    = task_weights,
            num_int_batches = num_int_batches)

    t1 = time.time()
    results_va = sc.evaluate_binary(net, loader_va, loss, dev)
    t2 = time.time()
    results_tr = sc.evaluate_binary(net, loader_tr, loss, dev)

    metrics_tr = results_tr['metrics'].loc[auc_cols].mean(0)
    metrics_va = results_va['metrics'].loc[auc_cols].mean(0)

    metrics_tr["epoch_time"] = t1 - t0
    metrics_va["epoch_time"] = t2 - t1

    if epoch % 20 == 0:
        print("Epoch\tlogl_tr  logl_va |  auc_tr   auc_va | aucpr_tr  aucpr_va | maxf1_tr  maxf1_va | tr_time")
    output_fstr = (
        f"{epoch}.\t{results_tr['logloss']:.5f}  {results_va['logloss']:.5f}"
        f" | {metrics_tr['roc_auc_score']:.5f}  {metrics_va['roc_auc_score']:.5f}"
        f" |  {metrics_tr['auc_pr']:.5f}   {metrics_va['auc_pr']:.5f}"
        f" |  {metrics_tr['max_f1_score']:.5f}   {metrics_va['max_f1_score']:.5f}"
        f" | {t1 - t0:6.1f}"
    )
    print(output_fstr)
    for metric_tr_name in metrics_tr.index:
        writer.add_scalar(metric_tr_name+"/tr", metrics_tr[metric_tr_name], epoch)
        writer.add_scalar(metric_tr_name+"/va", metrics_va[metric_tr_name], epoch)
    writer.add_scalar('logloss/tr', results_tr['logloss'], epoch)
    writer.add_scalar('logloss/va', results_va['logloss'], epoch)
    scheduler.step()

writer.close()
print("Saving performance metrics (AUCs) and model.")

if not os.path.exists("results"):
    os.makedirs("results")

aucs = pd.DataFrame({
    "num_pos": num_pos,
    "num_neg": num_neg,
    "num_pos_va": num_pos_va,
    "num_neg_va": num_neg_va,
    "auc_tr":  results_tr["metrics"]['roc_auc_score'],
    "auc_va":  results_va["metrics"]['roc_auc_score'],
    "auc_pr_tr":   results_tr["metrics"]["auc_pr"],
    "auc_pr_va":   results_va["metrics"]["auc_pr"],
    "avg_prec_tr": results_tr["metrics"]["avg_prec_score"],
    "avg_prec_va": results_va["metrics"]["avg_prec_score"],
    "max_f1_tr":   results_tr["metrics"]["max_f1_score"],
    "max_f1_va":   results_va["metrics"]["max_f1_score"],
})

aucs_file = f"results/{name}-metrics.csv"
aucs.to_csv(aucs_file)
print(f"Saved metrics (AUC, AUC-PR, MaxF1) for each task into '{aucs_file}'.")


#####   model saving   #####
if not os.path.exists("models"):
   os.makedirs("models")

model_file = f"models/{name}.pt"
conf_file  = f"models/{name}-conf.npy"

if args.save_model == 1 :
   torch.save(net.state_dict(), model_file)
   print(f"Saved model weights into '{model_file}'.")

results = {}
results["conf"] = args
results["results"] = {}
results["results"]["va"] = {"auc_roc": aucs["auc_va"], "auc_pr": aucs["auc_pr_va"], "logloss": results_va['logloss']}
results["results"]["tr"] = {"auc_roc": aucs["auc_tr"], "auc_pr": aucs["auc_pr_tr"], "logloss": results_tr['logloss']}
results["results_agg"]   = {"va": metrics_va, "tr": metrics_tr}

np.save(conf_file, results)
print(f"Saved config and results into '{conf_file}'.")
