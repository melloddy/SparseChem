import sklearn.metrics
import tqdm
import pandas as pd
import numpy as np
import torch

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

def evaluate_binary(net, loader, loss, dev):
    net.eval()
    logloss_sum   = 0.0
    logloss_count = 0
    y_ind_list    = []
    y_true_list   = []
    y_hat_list    = []

    with torch.no_grad():
        for b in tqdm.tqdm(loader, leave=False):
            X = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
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
