import sparsechem as sc
import scipy.io
import numpy as np
import pandas as pd
import torch
import tqdm
import argparse
import os
import os.path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--x", help="Descriptor file (matrix market)", type=str, default="chembl_23_x.mtx")
parser.add_argument("--y", help="Activity file (matrix market)", type=str, default="chembl_23_y.mtx")
parser.add_argument("--folding", help="Folding file (npy)", type=str, default="folding_hier_0.6.npy")
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.02)
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

args = parser.parse_args()

print(args)
if args.filename is not None:
    name = args.filename
else:
    name = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo{args.last_dropout:.1f}_wd{args.weight_decay}"
print(f"Run name is '{name}'.")

tb_name = "runs/"+name
writer = SummaryWriter(tb_name)
assert args.input_size_freq is None, "Using tail compression not yet supported."

ecfp    = scipy.io.mmread(args.x).tocsr()
ic50    = scipy.io.mmread(args.y).tocsr()
folding = np.load(args.folding)

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

fold_va = args.fold_va
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

batch_ratio = args.batch_ratio
batch_size  = int(np.ceil(batch_ratio * idx_tr.shape[0]))

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

for epoch in range(args.epochs):
    net.train()

    loss_sum   = 0.0
    loss_count = 0

    for b in tqdm.tqdm(loader_tr, leave=False):
        optimizer.zero_grad()
        X      = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], dataset_tr.input_size]).to(dev)
        y_ind  = b["y_ind"].to(dev)
        y_data = b["y_data"].to(dev)
        y_data = (y_data + 1) / 2.0

        yhat_all = net(X)
        yhat     = yhat_all[y_ind[0], y_ind[1]]
        
        output   = loss(yhat, y_data).sum()
        output_n = output / b["batch_size"]

        output_n.backward()

        optimizer.step()

        loss_sum   += output.detach() / y_data.shape[0]
        loss_count += 1

    results_va = sc.evaluate_binary(net, loader_va, loss, dev)
    results_tr = sc.evaluate_binary(net, loader_tr, loss, dev)

    loss_tr = loss_sum / loss_count
    metrics_tr = results_tr['metrics'].loc[auc_cols].mean(0)
    metrics_va = results_va['metrics'].loc[auc_cols].mean(0)

    if epoch % 20 == 0:
        print("Epoch\tlogl_tr  logl_va |  auc_tr   auc_va | aucpr_tr  aucpr_va | maxf1_tr  maxf1_va ")
    output_fstr = (
        f"{epoch}.\t{results_tr['logloss']:.5f}  {results_va['logloss']:.5f}"
        f" | {metrics_tr['roc_auc_score']:.5f}  {metrics_va['roc_auc_score']:.5f}"
        f" |  {metrics_tr['auc_pr']:.5f}   {metrics_va['auc_pr']:.5f}"
        f" |  {metrics_tr['max_f1_score']:.5f}   {metrics_va['max_f1_score']:.5f}"
    )
    print(output_fstr)
    for metric_tr_name in metrics_tr.index:
        #output_fstr = f"{output_fstr}\t{metric_tr_name}_tr = {metrics_tr[metric_tr_name]:.5f}\t{metric_tr_name}_va = {metrics_va[metric_tr_name]:.5f}"
        writer.add_scalar(metric_tr_name+"/tr", metrics_tr[metric_tr_name], epoch)
        writer.add_scalar(metric_tr_name+"/va", metrics_va[metric_tr_name], epoch)
    writer.add_scalar('logloss/tr', results_tr['logloss'], epoch)
    writer.add_scalar('logloss/va', results_va['logloss'], epoch)
    scheduler.step()

print("Saving performance metrics (AUCs) and model.")

if not os.path.exists("results"):
    os.makedirs("results")

aucs = pd.DataFrame({
    "num_pos": num_pos,
    "num_neg": num_neg,
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
model_file = f"models/{name}.pt"
conf_file  = f"models/{name}-conf.npy"

if not os.path.exists("models"):
    os.makedirs("models")

torch.save(net.state_dict(), model_file)
print(f"Saved model weights into '{model_file}'.")

results = {}
results["conf"] = args
results["results"] = {}
results["results"]["va"] = {"aucs": aucs["auc_va"], "logloss": results_va['logloss']}
results["results"]["tr"] = {"aucs": aucs["auc_tr"], "logloss": results_tr['logloss']}

np.save(conf_file, results)
print(f"Saved model conf into '{conf_file}'.")
