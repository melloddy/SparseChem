import sparsechem as sc
import scipy.io
import numpy as np
import pandas as pd
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--x", help="Descriptor file (matrix market)", type=str, default="chembl_23_x.mtx")
parser.add_argument("--y", help="Activity file (matrix market)", type=str, default="chembl_23_y.mtx")
parser.add_argument("--folding", help="Folding file (npy)", type=str, default="folding_hier_0.6.npy")
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.02)
parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes", default=[], type=int, required=True)
parser.add_argument("--middle_dropout", help="Last dropout", type=float, default=0.0)
parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
parser.add_argument("--last_non_linearity", help="Last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
parser.add_argument("--non_linearity", help="Before last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])
parser.add_argument("--input_size_freq", help="Number of high importance features", type=int, default=None)
parser.add_argument("--fold_inputs", help="Fold input to a fixed set (default no folding)", type=int, default=None)
parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
parser.add_argument("--min_samples_auc", help="Minimum number samples for AUC calculation", type=int, default=25)
parser.add_argument("--dev", help="Device to use", type=str, default="cuda:0")

args = parser.parse_args()

print(args)
name = f"sc_chembl_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo{args.last_dropout:.1f}_wd{args.weight_decay}"
print(f"Run name is '{name}'.")

assert args.input_size_freq is None, "Using tail compression not yet supported."

ecfp    = scipy.io.mmread(args.x).tocsr()
ic50    = scipy.io.mmread(args.y).tocsr()
folding = np.load(args.folding)

assert ecfp.shape[0] == ic50.shape[0]
assert ecfp.shape[0] == folding.shape[0]

if args.fold_inputs is not None:
    ecfp = sc.fold_inputs(ecfp, folding_size=args.fold_inputs)
    print(f"Folding inputs to {ecfp.shape[1]} dimensions.")

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

    for b in tqdm.tqdm(loader_tr):
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
    aucs_tr = results_tr["aucs"].loc[auc_cols].mean()
    aucs_va = results_va["aucs"].loc[auc_cols].mean()
    print(f"Epoch {epoch}.\tloss_tr_live={loss_tr:.5f}\tloss_tr={results_tr['logloss']:.5f}\tloss_va={results_va['logloss']:.5f}\taucs_tr={aucs_tr:.5f}\taucs_va={aucs_va:.5f}")
    scheduler.step()

results_file = f"{name}.npy"
results = {}
results["conf"] = args
results["results"] = {}
results["results"]["va"] = {"aucs": aucs_va, "logloss": results_va['logloss']}
results["results"]["tr"] = {"aucs": aucs_tr, "logloss": results_tr['logloss']}

np.save(results_file, results)
print(f"Saved results into '{results_file}'.")

aucs = pd.DataFrame({
    "num_pos": num_pos,
    "num_neg": num_neg,
    "auc_tr":  results_tr["aucs"],
    "auc_va":  results_va["aucs"],
})

aucs_file = f"{name}-aucs.csv"
aucs.to_csv(aucs_file)
print(f"Saved metrics for each task into '{aucs_file}'.")

