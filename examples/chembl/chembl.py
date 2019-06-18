import sparsechem as sc
import scipy.io
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

def auc_roc(y_true, y_score):
    if len(y_true) <= 1:
        return np.nan
    if (y_true[0] == y_true).all():
        return np.nan
    return sklearn.metrics.roc_auc_score(
          y_true  = y_true,
          y_score = y_score)

def compute_aucs(cols, y_true, y_score):
    df   = pd.DataFrame({"col": cols, "y_true": y_true, "y_score": y_score})
    aucs = df.groupby("col", sort=True).apply(lambda g:
              auc_roc(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values))
    return aucs

def evaluate(net, loader, loss, dev):
    net.eval()
    logloss_sum   = 0.0
    logloss_count = 0
    y_ind_list    = []
    y_true_list   = []
    y_hat_list    = []

    with torch.no_grad():
        for b in tqdm.tqdm(loader, leave=False):
            X = torch.sparse_coo_tensor(
                    b["x_ind"].to(dev),
                    b["x_data"].to(dev),
                    size = [b["batch_size"], loader.dataset.input_size])
            y_ind  = b["y_ind"].to(dev)
            y_data = b["y_data"].to(dev)
            y_data = (y_data + 1) / 2.0

            y_hat_all = net(X)
            y_hat     = y_hat_all[y_ind[0], y_ind[1]]
            output    = loss(y_hat, y_data).sum()
            logloss_sum   += output
            logloss_count += y_data.shape[0]

            ## storing data for AUCs
            y_ind_list.append(y_ind)
            y_true_list.append(y_data)
            y_hat_list.append(y_hat)

        y_ind  = torch.cat(y_ind_list, dim=1).cpu().numpy()
        y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
        y_hat  = torch.cat(y_hat_list, dim=0).cpu().numpy()
        aucs = compute_aucs(y_ind[1], y_true=y_true, y_score=y_hat)

        return {
            'aucs':    aucs,
            'logloss': logloss_sum / logloss_count
        }


ecfp = scipy.io.mmread("chembl_23_x.mtx").tocsr()
ic50 = scipy.io.mmread("chembl_23_y.mtx").tocsr()

num_pos  = np.array((ic50 == +1).sum(0)).flatten()
num_neg  = np.array((ic50 == -1).sum(0)).flatten()
auc_cols = np.where((num_pos >= 50) & (num_neg >= 50))[0]

print(f"There are {len(auc_cols)} columns for AUC calculation.")

## changing ic50 to 0/1 encoding
#ic50.data = (ic50.data == 1).astype(np.float32)

batch_size = int(np.ceil(0.02 * 0.8 * ecfp.shape[0]))
rperm      = np.random.permutation(ecfp.shape[0])
num_tr     = int(0.8 * ecfp.shape[0])
idx_tr     = rperm[:num_tr]
idx_va     = rperm[num_tr:]

dataset_tr = sc.SparseDataset(x=ecfp[idx_tr], y=ic50[idx_tr])
dataset_va = sc.SparseDataset(x=ecfp[idx_va], y=ic50[idx_va])

loader_tr  = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)
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

    results_va = evaluate(net, loader_va, loss, dev)
    results_tr = evaluate(net, loader_tr, loss, dev)

    loss_tr = loss_sum / loss_count
    aucs_tr = results_tr["aucs"].loc[auc_cols].mean()
    aucs_va = results_va["aucs"].loc[auc_cols].mean()
    print(f"Epoch {epoch}.\tloss_tr_live={loss_tr:.5f}\tloss_tr={results_tr['logloss']:.5f}\tloss_va={results_va['logloss']:.5f}\taucs_tr={aucs_tr:.5f}\taucs_va={aucs_va:.5f}")


