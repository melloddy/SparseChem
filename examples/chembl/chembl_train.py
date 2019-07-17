import sparsechem as sc
import scipy.io
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

ecfp    = scipy.io.mmread("chembl_23_x.mtx").tocsr()
ic50    = scipy.io.mmread("chembl_23_y.mtx").tocsr()
folding = np.load("./folding_hier_0.6.npy")

num_pos  = np.array((ic50 == +1).sum(0)).flatten()
num_neg  = np.array((ic50 == -1).sum(0)).flatten()
auc_cols = np.where((num_pos >= 50) & (num_neg >= 50))[0]

print(f"There are {len(auc_cols)} columns for AUC calculation (i.e., at least 50 positives and 50 negatives).")

fold_va = 0
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

batch_ratio = 0.02
batch_size  = int(np.ceil(batch_ratio * idx_tr.shape[0]))

dataset_tr = sc.SparseDataset(x=ecfp[idx_tr], y=ic50[idx_tr])
dataset_va = sc.SparseDataset(x=ecfp[idx_va], y=ic50[idx_va])

loader_tr  = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=sc.sparse_collate)
loader_va  = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

conf = sc.ModelConfig(
    input_size         = dataset_tr.input_size,
    hidden_sizes       = [400],
    output_size        = dataset_tr.output_size,
    #input_size_freq    = 1000,
    #tail_hidden_size   = 20,
    last_dropout       = 0.2,
    weight_decay       = 1e-5,
    non_linearity      = "relu",
    last_non_linearity = "relu",
)
## custom conf options
conf.lr       = 1e-3
conf.lr_alpha = 0.3
conf.lr_steps = [10]
conf.epochs   = 20

dev  = "cuda:0"
net  = sc.SparseFFN(conf).to(dev)
loss = torch.nn.BCEWithLogitsLoss(reduction="none")

optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=conf.lr_steps, gamma=0.3)

for epoch in range(conf.epochs):
    scheduler.step()
    net.train()

    loss_sum   = 0.0
    loss_count = 0

    for b in tqdm.tqdm(loader_tr):
        optimizer.zero_grad()
        X      = torch.sparse_coo_tensor(
                    b["x_ind"].to(dev),
                    b["x_data"].to(dev),
                    size = [b["batch_size"], dataset_tr.input_size])
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


