# Copyright (c) 2020 KU Leuven
import sklearn.metrics
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import scipy.sparse
import scipy.io
import scipy.special
import types
import json
import warnings
from sparsechem import censored_mse_loss_numpy
from collections import namedtuple

class Nothing(object):
    def __getattr__(self, name):
        return Nothing()
    def __call__(self, *args, **kwargs):
        return Nothing()
    def __repr__(self):
        return "Nothing"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_acc_kappa(recall, fpr, num_pos, num_neg):
    """Calculates accuracy from recall and precision."""
    num_all = num_neg + num_pos
    tp = np.round(recall * num_pos).astype(np.int)
    fn = num_pos - tp
    fp = np.round(fpr * num_neg).astype(np.int)
    tn = num_neg - fp
    acc   = (tp + tn) / num_all
    pexp  = num_pos / num_all * (tp + fp) / num_all + num_neg / num_all * (tn + fn) / num_all
    kappa = (acc - pexp) / (1 - pexp)
    return acc, kappa

def all_metrics(y_true, y_score):
    """Compute classification metrics.
    Args:
        y_true     true labels (0 / 1)
        y_score    logit values
    """
    if len(y_true) <= 1 or (y_true[0] == y_true).all():
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "f1_max": [np.nan], "p_f1_max": [np.nan], "kappa": [np.nan], "kappa_max": [np.nan], "p_kappa_max": [np.nan]})
        return df

    fpr, tpr, tpr_thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
    roc_auc_score = sklearn.metrics.auc(x=fpr, y=tpr)
    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    ## calculating F1 for all cutoffs
    F1_score       = np.zeros(len(precision))
    mask           = precision > 0
    F1_score[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
    f1_max_idx     = F1_score.argmax()
    f1_max         = F1_score[f1_max_idx]
    p_f1_max       = scipy.special.expit(pr_thresholds[f1_max_idx])

    auc_pr = sklearn.metrics.auc(x = recall, y = precision)
    avg_prec_score = sklearn.metrics.average_precision_score(
          y_true  = y_true,
          y_score = y_score)
    y_classes = np.where(y_score >= 0.0, 1, 0)

    ## accuracy for all thresholds
    acc, kappas   = calc_acc_kappa(recall=tpr, fpr=fpr, num_pos=(y_true==1).sum(), num_neg=(y_true==0).sum())
    kappa_max_idx = kappas.argmax()
    kappa_max     = kappas[kappa_max_idx]
    p_kappa_max   = scipy.special.expit(tpr_thresholds[kappa_max_idx])

    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_classes)
    df = pd.DataFrame({"roc_auc_score": [roc_auc_score], "auc_pr": [auc_pr], "avg_prec_score": [avg_prec_score], "f1_max": [f1_max], "p_f1_max": [p_f1_max], "kappa": [kappa], "kappa_max": [kappa_max], "p_kappa_max": p_kappa_max})
    return df

def compute_corr(x, y):
    if len(y) <= 1:
        return np.nan
    ystd = y.std()
    xstd = x.std()
    if ystd == 0 or xstd == 0:
        return np.nan
    return np.dot((x - x.mean()), (y - y.mean())) / len(y) / y.std() / x.std()

def all_metrics_regr(y_true, y_score, y_censor=None):
    if len(y_true) <= 1:
        df = pd.DataFrame({"rmse": [np.nan], "rmse_uncen": [np.nan], "rsquared": [np.nan], "corrcoef": [np.nan]})
        return df
    ## censor0 means non-censored observations
    censor0 = y_censor == 0 if y_censor is not None else slice(None)
    mse_cen = censored_mse_loss_numpy(target=y_true, input=y_score, censor=y_censor).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mse  = ((y_true[censor0] - y_score[censor0])**2).mean()
        yvar = y_true[censor0].var()
    if yvar == 0 or np.isnan(yvar):
        rsquared = np.nan
        corr = np.nan
    else:
        rsquared = 1 - mse / yvar
        corr     = compute_corr(y_true[censor0], y_score[censor0])
    df = pd.DataFrame({
        "rmse":       [np.sqrt(mse_cen)],
        "rmse_uncen": [np.sqrt(mse)],
        "rsquared":   [rsquared],
        "corrcoef":   [corr],
    })
    return df

def compute_metrics(cols, y_true, y_score, num_tasks):
    if len(cols) < 1:
        return pd.DataFrame({
            "roc_auc_score": np.nan,
            "auc_pr": np.nan,
            "avg_prec_score": np.nan,
            "f1_max": np.nan,
            "p_f1_max": np.nan,
            "kappa": np.nan,
            "kappa_max": np.nan,
            "p_kappa_max": np.nan}, index=np.arange(num_tasks))
    df   = pd.DataFrame({"task": cols, "y_true": y_true, "y_score": y_score})
    metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values))
    metrics.reset_index(level=-1, drop=True, inplace=True)
    return metrics.reindex(np.arange(num_tasks))

def compute_metrics_regr(cols, y_true, y_score, num_tasks, y_censor=None):
    """Returns metrics for regression tasks."""
    if len(cols) < 1:
        return pd.DataFrame({
            "rmse": np.nan,
            "rsquared": np.nan,
            "corrcoef": np.nan,
            },
            index=np.arange(num_tasks))
    df = pd.DataFrame({
        "task": cols,
        "y_true": y_true,
        "y_score": y_score,
        "y_censor": y_censor,
    })
    metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics_regr(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values,
                  y_censor = g.y_censor.values if y_censor is not None else None))
    metrics.reset_index(level=-1, drop=True, inplace=True)
    return metrics.reindex(np.arange(num_tasks))

def class_fold_counts(y_class, folding):
    folds = np.unique(folding)
    num_pos = []
    num_neg = []
    for fold in folds:
        yf = y_class[folding == fold]
        num_pos.append( np.array((yf == +1).sum(0)).flatten() )
        num_neg.append( np.array((yf == -1).sum(0)).flatten() )
    return np.row_stack(num_pos), np.row_stack(num_neg)


def print_metrics(epoch, train_time, metrics_tr, metrics_va, header):
    if metrics_tr is None:
        if header:
            print("Epoch\tlogl_va |  auc_va | aucpr_va | maxf1_va | tr_time")
        output_fstr = (
            f"{epoch}.\t{metrics_va['logloss']:.5f}"
            f" | {metrics_va['roc_auc_score']:.5f}"
            f" |  {metrics_va['auc_pr']:.5f}"
            f" |  {metrics_va['f1_max']:.5f}"
            f" | {train_time:6.1f}"
        )
        print(output_fstr)
        return

    ## full print
    if header:
        print("Epoch\tlogl_tr  logl_va |  auc_tr   auc_va | aucpr_tr  aucpr_va | maxf1_tr  maxf1_va | tr_time")
    output_fstr = (
        f"{epoch}.\t{metrics_tr['logloss']:.5f}  {metrics_va['logloss']:.5f}"
        f" | {metrics_tr['roc_auc_score']:.5f}  {metrics_va['roc_auc_score']:.5f}"
        f" |  {metrics_tr['auc_pr']:.5f}   {metrics_va['auc_pr']:.5f}"
        f" |  {metrics_tr['f1_max']:.5f}   {metrics_va['f1_max']:.5f}"
        f" | {train_time:6.1f}"
    )
    print(output_fstr)

def print_table(formats, data):
    for key, fmt in formats.items():
        print(fmt.format(data[key]), end="")

Column = namedtuple("Column", "key size dec title")
columns_cr = [
    Column("epoch",         size=6, dec= 0, title="Epoch"),
    Column(None,            size=1, dec=-1, title="|"),
    Column("logloss",       size=8, dec= 5, title="logl"),
    Column("roc_auc_score", size=8, dec= 5, title="aucroc"),
    Column("auc_pr",        size=8, dec= 5, title="aucpr"),
    Column("f1_max",        size=8, dec= 5, title="f1_max"),
    Column(None,            size=1, dec=-1, title="|"),
    Column("rmse",          size=9, dec= 5, title="rmse"),
    Column("rsquared",      size=9, dec= 5, title="rsquared"),
    Column("corrcoef",      size=9, dec= 5, title="corrcoef"),
    Column(None,            size=1, dec=-1, title="|"),
    Column("train_time",    size=6, dec= 1, title="tr_time"),
]

def print_cell(value, size, dec, left, end=" "):
    align = "<" if left else ">"
    if type(value) == str:
        print(("{:" + align + str(size) + "}").format(value), end=end)
    else:
        print(("{:" + align + str(size) + "." + str(dec) + "f}").format(value), end=end)

def print_metrics_cr(epoch, train_time, results_tr, results_va, header):
    data = pd.concat([results_va["classification_agg"], results_va["regression_agg"]])
    data["train_time"] = train_time
    data["epoch"] = epoch
    if header:
        for i, col in enumerate(columns_cr):
            print_cell(col.title, col.size, dec=0, left=(i==0))
        print()
    ## printing row with values
    for i, col in enumerate(columns_cr):
        print_cell(data.get(col.key, col.title), col.size, dec=col.dec, left=(i==0))
    print()

def evaluate_binary(net, loader, loss, dev, progress=True):
    net.eval()
    logloss_sum   = 0.0
    logloss_count = 0
    y_ind_list    = []
    y_true_list   = []
    y_hat_list    = []
    num_tasks     = loader.dataset.y.shape[1]

    with torch.no_grad():
        for b in tqdm(loader, leave=False, disable=(progress == False)):
            X = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
            y_ind  = b["y_ind"].to(dev)
            y_data = b["y_data"].to(dev)

            y_hat_all = net(X)
            y_hat     = y_hat_all[y_ind[0], y_ind[1]]
            output    = loss(y_hat, y_data).sum()
            logloss_sum   += output
            logloss_count += y_data.shape[0]

            ## storing data for AUCs
            y_ind_list.append(y_ind.cpu())
            y_true_list.append(y_data.cpu())
            y_hat_list.append(y_hat.cpu())

        if len(y_ind_list) == 0:
            return {
                "metrics": compute_metrics([], y_true=[], y_score=[], num_tasks=num_tasks),
                "logloss": np.nan,
            }
        y_ind  = torch.cat(y_ind_list, dim=1).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_hat  = torch.cat(y_hat_list, dim=0).numpy()
        metrics = compute_metrics(y_ind[1], y_true=y_true, y_score=y_hat, num_tasks=num_tasks)

        return {
            'metrics': metrics,
            'logloss': logloss_sum.cpu().numpy() / logloss_count
        }

def train_binary(net, optimizer, loader, loss, dev, task_weights, normalize_loss=None, num_int_batches=1, progress=True):
    """
    Args:
        net         pytorch network
        optimizer   optimizer to use
        loader      data loader with training data
        dev         device
        task_weights     weights of the tasks
        normalize_loss   normalization value, if None then use batch size 
        num_int_batches  number of internal batches to use
        progress         whether to show a progress bar
    """
    net.train()
    logloss_sum   = 0.0
    logloss_count = 0

    int_count = 0
    for b in tqdm(loader, leave=False, disable=(progress == False)):
        if int_count == 0:
            optimizer.zero_grad()

        X       = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
        y_ind   = b["y_ind"].to(dev)
        y_w     = task_weights[y_ind[1]]
        y_data  = b["y_data"].to(dev)

        yhat_all = net(X)
        yhat     = yhat_all[y_ind[0], y_ind[1]]

        norm = normalize_loss
        if norm is None:
            norm = b["batch_size"] * num_int_batches
        
        output   = (loss(yhat, y_data) * y_w).sum()
        output_n = output / norm
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

def batch_forward(net, b, input_size, loss_class, loss_regr, weights_class, weights_regr, dev):
    """returns full outputs from the network for the batch b"""
    X = torch.sparse_coo_tensor(
        b["x_ind"],
        b["x_data"],
        size = [b["batch_size"], input_size]).to(dev, non_blocking=True)

    yc_hat_all, yr_hat_all = net(X)

    out = {}
    out["yc_hat_all"] = yc_hat_all
    out["yr_hat_all"] = yr_hat_all
    out["yc_loss"]    = 0
    out["yr_loss"]    = 0
    out["yc_weights"] = 0
    out["yr_weights"] = 0

    if out["yc_hat_all"] is not None:
        yc_ind  = b["yc_ind"].to(dev, non_blocking=True)
        yc_w    = weights_class[yc_ind[1]]
        yc_data = b["yc_data"].to(dev, non_blocking=True)
        yc_hat  = yc_hat_all[yc_ind[0], yc_ind[1]]
        out["yc_ind"]  = yc_ind
        out["yc_data"] = yc_data
        out["yc_hat"]  = yc_hat
        out["yc_loss"] = (loss_class(yc_hat, yc_data) * yc_w).sum()
        out["yc_weights"] = yc_w.sum()

    if out["yr_hat_all"] is not None:
        yr_ind  = b["yr_ind"].to(dev, non_blocking=True)
        yr_w    = weights_regr[yr_ind[1]]
        yr_data = b["yr_data"].to(dev, non_blocking=True)
        yr_hat  = yr_hat_all[yr_ind[0], yr_ind[1]]

        out["ycen_data"] = b["ycen_data"]
        if out["ycen_data"] is not None:
            out["ycen_data"] = out["ycen_data"].to(dev, non_blocking=True)

        out["yr_ind"]  = yr_ind
        out["yr_data"] = yr_data
        out["yr_hat"]  = yr_hat
        out["yr_loss"] = (loss_regr(input=yr_hat, target=yr_data, censor=out["ycen_data"]) * yr_w).sum()
        out["yr_weights"] = yr_w.sum()

    return out


def train_class_regr(net, optimizer, loader, loss_class, loss_regr, dev,
                     weights_class, weights_regr,
                     normalize_loss=None, num_int_batches=1, progress=True):
    net.train()

    int_count = 0
    for b in tqdm(loader, leave=False, disable=(progress == False)):
        if int_count == 0:
            optimizer.zero_grad()

        norm = normalize_loss
        if norm is None:
            norm = b["batch_size"] * num_int_batches

        fwd = batch_forward(net, b=b, input_size=loader.dataset.input_size, loss_class=loss_class, loss_regr=loss_regr, weights_class=weights_class, weights_regr=weights_regr, dev=dev)
        loss = fwd["yc_loss"] + fwd["yr_loss"]
        loss_norm = loss / norm
        loss_norm.backward()

        int_count += 1
        if int_count == num_int_batches:
            optimizer.step()
            int_count = 0

    if int_count > 0:
        ## process tail batch (should not happen)
        optimizer.step()


def aggregate_results(df, weights):
    """Compute aggregates based on the weights"""
    wsum = weights.sum()
    if wsum == 0:
        return pd.Series(np.nan, index=df.columns)
    wnorm = weights / wsum
    return (df * wnorm[:,None]).reindex(labels=np.where(weights > 0)[0]).sum(0, skipna=False)


def evaluate_class_regr(net, loader, loss_class, loss_regr, tasks_class, tasks_regr, dev, progress=True):
    class_w = tasks_class.aggregation_weight
    regr_w  = tasks_regr.aggregation_weight

    net.eval()
    loss_class_sum   = 0.0
    loss_regr_sum    = 0.0
    loss_class_weights = 0.0
    loss_regr_weights  = 0.0
    data = {
        "yc_ind":  [],
        "yc_data": [],
        "yc_hat":  [],
        "yr_ind":  [],
        "yr_data": [],
        "yr_hat":  [],
        "ycen_data": [],
    }
    num_class_tasks  = loader.dataset.class_output_size
    num_regr_tasks   = loader.dataset.regr_output_size

    with torch.no_grad():
        for b in tqdm(loader, leave=False, disable=(progress == False)):
            fwd = batch_forward(net, b=b, input_size=loader.dataset.input_size, loss_class=loss_class, loss_regr=loss_regr, weights_class=tasks_class.training_weight, weights_regr=tasks_regr.training_weight, dev=dev)
            loss_class_sum += fwd["yc_loss"]
            loss_regr_sum  += fwd["yr_loss"]
            loss_class_weights += fwd["yc_weights"]
            loss_regr_weights  += fwd["yr_weights"]

            ## storing data for AUCs
            for key in data.keys():
                if (key in fwd) and (fwd[key] is not None):
                    data[key].append(fwd[key].cpu())

        out = {}
        if len(data["yc_ind"]) == 0:
            ## there are no data for classification
            out["classification"] = compute_metrics([], y_true=[], y_score=[], num_tasks=num_class_tasks)
            out["classification_agg"] = out["classification"].reindex(labels=[]).mean(0)
            out["classification_agg"]["logloss"] = np.nan
        else:
            yc_ind  = torch.cat(data["yc_ind"], dim=1).numpy()
            yc_data = torch.cat(data["yc_data"], dim=0).numpy()
            yc_hat  = torch.cat(data["yc_hat"], dim=0).numpy()
            out["classification"] = compute_metrics(yc_ind[1], y_true=yc_data, y_score=yc_hat, num_tasks=num_class_tasks)
            out["classification_agg"] = aggregate_results(out["classification"], weights=class_w)
            out["classification_agg"]["logloss"] = loss_class_sum.cpu().item() / loss_class_weights.cpu().item()

        if len(data["yr_ind"]) == 0:
            out["regression"] = compute_metrics_regr([], y_true=[], y_score=[], num_tasks=num_regr_tasks)
            out["regression_agg"] = out["regression"].reindex(labels=[]).mean(0)
            out["regression_agg"]["mseloss"] = np.nan
        else:
            yr_ind  = torch.cat(data["yr_ind"], dim=1).numpy()
            yr_data = torch.cat(data["yr_data"], dim=0).numpy()
            yr_hat  = torch.cat(data["yr_hat"], dim=0).numpy()
            if len(data["ycen_data"]) > 0:
                ycen_data = torch.cat(data["ycen_data"], dim=0).numpy()
            else:
                ycen_data = None
            out["regression"] = compute_metrics_regr(yr_ind[1], y_true=yr_data, y_score=yr_hat, y_censor=ycen_data, num_tasks=num_regr_tasks)
            out["regression_agg"] = aggregate_results(out["regression"], weights=regr_w)
            out["regression_agg"]["mseloss"] = loss_regr_sum.cpu().item() / loss_regr_weights.cpu().item()

        out["classification_agg"]["num_tasks_total"] = loader.dataset.class_output_size
        out["classification_agg"]["num_tasks_agg"]   = (tasks_class.aggregation_weight > 0).sum()
        out["regression_agg"]["num_tasks_total"] = loader.dataset.regr_output_size
        out["regression_agg"]["num_tasks_agg"]   = (tasks_regr.aggregation_weight > 0).sum()

        return out

def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def predict(net, loader, dev, last_hidden=False, progress=True, dropout=False):
    """
    Makes predictions for all compounds in the loader.
    """
    net.eval()
    if dropout:
        net.apply(enable_dropout)

    y_class_list = []
    y_regr_list  = []

    with torch.no_grad():
        for b in tqdm(loader, leave=False, disable=(progress == False)):
            X = torch.sparse_coo_tensor(
                    b["x_ind"],
                    b["x_data"],
                    size = [b["batch_size"], loader.dataset.input_size]).to(dev)
            y_class, y_regr = net(X, last_hidden=last_hidden)
            if net.class_output_size > 0:
                y_class_list.append(y_class.cpu())
            if net.regr_output_size > 0:
                y_regr_list.append(y_regr.cpu())

    y_class = None
    y_regr  = None
    if net.class_output_size > 0:
        y_class = torch.cat(y_class_list, dim=0)
    if net.regr_output_size > 0:
        y_regr  = torch.cat(y_regr_list, dim=0)
    return y_class, y_regr

def fold_transform_inputs(x, folding_size=None, transform="none"):
    """Fold and transform sparse matrix x:
    Args:
        x             sparse matrix
        folding_size  modulo for folding
        transform     "none", "binarize", "tanh", "log1p"
    Returns folded and transformed x.
    """
    if folding_size is not None and x.shape[1] > folding_size:
        ## collapse x into folding_size columns
        idx = x.nonzero()
        folded = idx[1] % folding_size
        x = scipy.sparse.csr_matrix((x.data, (idx[0], folded)), shape=(x.shape[0], folding_size))

    if transform is None or transform == "none":
        pass
    elif transform == "binarize":
        x.data = (x.data > 0).astype(np.float32)
    elif transform == "tanh":
        x.data = np.tanh(x.data)
    elif transform == "log1p":
        x.data = np.log1p(x.data)
    else:
        raise ValueError(f"Unknown input transformation '{transform}'.")
    return x

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
    if filename is None:
        return None
    if filename.endswith('.mtx'):
        return scipy.io.mmread(filename).tocsr()
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True).item().tocsr()
    raise ValueError(f"Loading '{filename}' failed. It must have a suffix '.mtx' or '.npy'.")

def load_task_weights(filename, y, label):
    """Loads and processes task weights, otherwise raises an error using the label.
    Args:
        df      DataFrame with weights
        y       csr matrix of labels
        label   name for error messages
    Returns tuple of
        training_weight
        aggregation_weight
        task_type
    """
    res = types.SimpleNamespace(training_weight=None, aggregation_weight=None, task_type=None)
    if y is None:
        assert filename is None, f"Weights provided for {label}, please add also --{label}"
        res.training_weight = torch.ones(0)
        return res

    if filename is None:
        res.training_weight = torch.ones(y.shape[1])
        return res

    df = pd.read_csv(filename)

    assert "task_id" in df.columns, "task_id is missing in task info CVS file"
    df.rename(columns={"weight": "training_weight"})
    assert "training_weight" in df.columns, "training_weight is missing in task info CSV file"
    df.sort_values("task_id", inplace=True)

    assert y.shape[1] == df.shape[0], f"task weights for '{label}' have different size ({df.shape[0]}) to {label} columns ({y.shape[1]})."
    assert (0 <= df.training_weight).all(), f"task weights (for {label}) must not be negative"
    assert (df.training_weight <= 1).all(), f"task weights (for {label}) must not be larger than 1.0"

    assert df.task_id.unique().shape[0] == df.shape[0], f"task ids (for {label}) are not all unique"
    assert (0 <= df.task_id).all(), f"task ids in task weights (for {label}) must not be negative"
    assert (df.task_id < df.shape[0]).all(), f"task ids in task weights (for {label}) must be below number of tasks"

    res.training_weight = torch.FloatTensor(df.training_weight.values)
    if "aggregation_weight" in df:
        assert (0 <= df.aggregation_weight).all(), f"Found negative aggregation_weight for {label}. Aggregation weights must be non-negative."
        res.aggregation_weight = df.aggregation_weight.values
    if "task_type" in df:
        res.task_type = df.task_type.values

    return res

def save_results(filename, conf, validation, training):
    """Saves conf and results into json file. Validation and training can be None."""
    out = {}
    out["conf"] = conf.__dict__

    if validation is not None:
        out["validation"] = {}
        for key in ["classification", "classification_agg", "regression", "regression_agg"]:
            out["validation"][key] = validation[key].to_json()

    if training is not None:
        out["training"] = {}
        for key in ["classification", "classification_agg", "regression", "regression_agg"]:
            out["training"][key] = training[key].to_json()
    with open(filename, "w") as f:
        json.dump(out, f)


def load_results(filename, two_heads=False):
    """Loads conf and results from a file
    Args:
        filename    name of the json/npy file
        two_heads   set up class_output_size if missing
    """
    if filename.endswith(".npy"):
        return np.load(filename, allow_pickle=True).item()

    with open(filename, "r") as f:
        data = json.load(f)

    for key in ["model_type"]:
        if key not in data["conf"]:
            data["conf"][key] = None
    if two_heads and ("class_output_size" not in data["conf"]):
        data["conf"]["class_output_size"] = data["conf"]["output_size"]
        data["conf"]["regr_output_size"]  = 0

    data["conf"] = types.SimpleNamespace(**data["conf"])


    if "results" in data:
        for key in data["results"]:
            data["results"][key] = pd.read_json(data["results"][key])

    if "results_agg" in data:
        for key in data["results_agg"]:
            data["results_agg"][key] = pd.read_json(data["results_agg"][key], typ="series")

    for key in ["training", "validation"]:
        if key not in data:
            continue
        for dfkey in ["classification", "regression"]:
            data[key][dfkey] = pd.read_json(data[key][dfkey])
        for skey in ["classification_agg", "regression_agg"]:
            data[key][skey]  = pd.read_json(data[key][skey], typ="series")

    return data
