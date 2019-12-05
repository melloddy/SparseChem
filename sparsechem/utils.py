import sklearn.metrics
import tqdm
import pandas as pd
import numpy as np
import torch
import scipy.sparse
import scipy.io

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def all_metrics(y_true, y_score):
    if len(y_true) <= 1:
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "max_f1_score": [np.nan]})
        return df
    if (y_true[0] == y_true).all():
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "max_f1_score": [np.nan]})
        return df
    roc_auc_score = sklearn.metrics.roc_auc_score(
          y_true  = y_true,
          y_score = y_score)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    ## calculating F1 for all cutoffs
    F1_score       = np.zeros(len(precision))
    mask           = precision > 0
    F1_score[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])

    max_f1_score = F1_score.max()
    auc_pr = sklearn.metrics.auc(x = recall, y = precision)
    avg_prec_score = sklearn.metrics.average_precision_score(
          y_true  = y_true,
          y_score = y_score)
    df = pd.DataFrame({"roc_auc_score": [roc_auc_score], "auc_pr": [auc_pr], "avg_prec_score": [avg_prec_score], "max_f1_score": [max_f1_score]})
    return df

def compute_metrics(cols, y_true, y_score):
    df   = pd.DataFrame({"task": cols, "y_true": y_true, "y_score": y_score})
    metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values))
    metrics.reset_index(level=-1, drop=True, inplace=True)
    return metrics

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
        metrics = compute_metrics(y_ind[1], y_true=y_true, y_score=y_hat)

        return {
            'metrics': metrics,
            'logloss': logloss_sum.cpu().numpy() / logloss_count
        }

def train_binary(net, optimizer, loader, loss, dev, task_weights, num_int_batches=1):
    net.train()
    logloss_sum   = 0.0
    logloss_count = 0

    int_count = 0
    for b in tqdm.tqdm(loader, leave=False):
        if int_count == 0:
            optimizer.zero_grad()

        X       = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
        y_ind   = b["y_ind"].to(dev)
        y_w     = task_weights[y_ind[1]]
        y_data  = b["y_data"].to(dev)
        y_data  = (y_data + 1) / 2.0

        yhat_all = net(X)
        yhat     = yhat_all[y_ind[0], y_ind[1]]
        
        output   = (loss(yhat, y_data) * y_w).sum()
        output_n = output / (b["batch_size"] * num_int_batches)

        output_n.backward()

        int_count += 1
        if int_count == num_int_batches:
            optimizer.step()
            int_count = 0

        logloss_sum   += output.detach() / y_data.shape[0]
        logloss_count += 1

    if int_count > 0:
        ## process tail batch (should not happen)
        optimizer.step()
    return logloss_sum / logloss_count

def predict(net, loader, dev):
    """
    Makes predictions for all compounds in the loader.
    """
    net.eval()

    y_hat_list = []

    with torch.no_grad():
        for b in tqdm.tqdm(loader, leave=False):
            X = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
            y_hat = net(X)
            y_hat_list.append(y_hat)

        y_hat = torch.cat(y_hat_list, dim=0)
        return y_hat

def fold_inputs(x, folding_size, binarize=True):
    if x.shape[1] <= folding_size:
        return x
    ## collapse x into folding_size columns
    idx = x.nonzero()
    folded = idx[1] % folding_size
    x2  = scipy.sparse.csr_matrix((x.data, (idx[0], folded)), shape=(x.shape[0], folding_size))
    if binarize:
        x2.data = (x2.data > 0).astype(np.float)
    return x2


def set_weights(net, filename="./tf_h400_inits.npy"):
    """
    Loads weights from disk and net parameters from them.
    """
    print(f"Loading weights from '{filename}'.")
    torch_to_value = np.load(filename, allow_pickle=True).item()
    for name, param in net.named_parameters():
        value = torch_to_value[name]
        if value.shape != param.shape:
            value = value.T
        assert value.shape == param.shape
        param.data.copy_(torch.FloatTensor(value))
    print("Weights have been copied to Pytorch net.")


def load_sparse(filename):
    """Loads sparse from Matrix market or Numpy .npy file."""
    if filename.endswith('.mtx'):
       return scipy.io.mmread(filename).tocsr()
    elif filename.endswith('.npy'):
       return np.load(filename, allow_pickle=True).item().tocsr()
    return None

