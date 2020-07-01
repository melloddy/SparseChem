# Copyright (c) 2020 KU Leuven
from torch.utils.data import Dataset
import torch
import scipy.sparse
import numpy as np

class SparseDataset(Dataset):
    def __init__(self, x, y):
        '''
        Args:
            X (sparse matrix):  input [n_sampes, features_in]
            Y (sparse matrix):  output [n_samples, features_out]
        '''
        assert x.shape[0]==y.shape[0], f"Input has {x.shape[0]} rows, output has {y.shape[0]} rows."

        self.x = x.tocsr(copy=False).astype(np.float32)
        self.y = y.tocsr(copy=False).astype(np.float32)
        # scale labels from {-1, +1} to {0, 1}, zeros are stored explicitly
        self.y.data = (self.y.data + 1) / 2.0

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        x_start = self.x.indptr[idx]
        x_end = self.x.indptr[idx + 1]
        x_indices = self.x.indices[x_start:x_end]
        x_data = self.x.data[x_start:x_end]

        y_start = self.y.indptr[idx]
        y_end = self.y.indptr[idx + 1]
        y_indices = self.y.indices[y_start:y_end]
        y_data = self.y.data[y_start:y_end]

        return {
            "x_ind":  x_indices,
            "x_data": x_data,
            "y_ind":  y_indices,
            "y_data": y_data,
        }

    def batch_to_x(self, batch, dev):
        """Takes 'xind' and 'x_data' from batch and converts them into a sparse tensor.
        Args:
            batch  batch
            dev    device to send the tensor to
        """
        return torch.sparse_coo_tensor(
                batch["x_ind"].to(dev),
                batch["x_data"].to(dev),
                size=[batch["batch_size"], self.x.shape[1]])


def sparse_collate(batch):
    x_ind  = [b["x_ind"]  for b in batch]
    x_data = [b["x_data"] for b in batch]
    y_ind  = [b["y_ind"]  for b in batch]
    y_data = [b["y_data"] for b in batch]

    ## x matrix
    xrow = np.repeat(np.arange(len(x_ind)), [len(i) for i in x_ind])
    xcol = np.concatenate(x_ind)
    xv   = np.concatenate(x_data)

    ## y matrix
    yrow  = np.repeat(np.arange(len(y_ind)), [len(i) for i in y_ind])
    ycol  = np.concatenate(y_ind).astype(np.int64)

    return {
        "x_ind":  torch.LongTensor([xrow, xcol]),
        "x_data": torch.from_numpy(xv),
        "y_ind":  torch.stack([torch.from_numpy(yrow), torch.from_numpy(ycol)], dim=0),
        "y_data": torch.from_numpy(np.concatenate(y_data)),
        "batch_size": len(batch),
    }

def get_row(csr, row):
    """returns row from csr matrix: indices and values."""
    start = csr.indptr[row]
    end   = csr.indptr[row + 1]
    return csr.indices[start:end], csr.data[start:end]

def to_idx_tensor(idx_list):
    """Turns list of lists [num_lists, 2] tensor of coordinates"""
    xrow = np.repeat(np.arange(len(idx_list)), [len(i) for i in idx_list])
    xcol = np.concatenate(idx_list)
    return torch.LongTensor([xrow, xcol])

def patterns_match(x, y):
    if y.shape != x.shape:             return False
    if y.nnz != x.nnz:                 return False
    if (y.indices != x.indices).any(): return False
    if (y.indptr != x.indptr).any():   return False
    return True

class ClassRegrSparseDataset(Dataset):
    def __init__(self, x, y_class, y_regr, y_censor=None):
        '''
        Creates dataset for two outputs Y.
        Args:
            x (sparse matrix):        input [n_sampes, features_in]
            y_class (sparse matrix):  class data [n_samples, class_tasks]
            y_regr (sparse matrix):   regression data [n_samples, regr_tasks]
            y_censor (sparse matrix): censoring matrix, for regression data [n_samples, regr_task]
        '''
        if y_censor is None:
            y_censor = scipy.sparse.csr_matrix(y_regr.shape)
        assert y_class.shape[1] + y_regr.shape[1] > 0, "No labels provided (both y_class and y_regr are missing)"
        assert x.shape[0]==y_class.shape[0], f"Input has {x.shape[0]} rows and class data {y_class.shape[0]} rows. Must be equal."
        assert x.shape[0]==y_regr.shape[0], f"Input has {x.shape[0]} rows and regression data has {y_regr.shape[0]} rows. Must be equal."
        assert y_regr.shape==y_censor.shape, f"Regression data has shape {y_regr.shape} and censor data has shape {y_censor.shape[0]}. Must be equal."

        self.x       = x.tocsr(copy=False).astype(np.float32)
        self.y_class = y_class.tocsr(copy=False).astype(np.float32)
        self.y_regr  = y_regr.tocsr(copy=False).astype(np.float32)
        self.y_censor = y_censor.tocsr(copy=False).astype(np.float32)

        if self.y_censor.nnz > 0:
            assert patterns_match(self.y_regr, self.y_censor), "y_regr and y_censor must have the same shape and sparsity pattern (nnz, indices and indptr)"
            d = self.y_censor.data
            assert ((d == -1) | (d == 0) | (d == 1)).all(), "Values of regression censor (y_censor) must be either -1, 0 or 1."

        # scale labels from {-1, +1} to {0, 1}, zeros are stored explicitly
        self.y_class.data = (self.y_class.data + 1) / 2.0

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y_class.shape[1] + self.y_regr.shape[1]

    @property
    def class_output_size(self):
        return self.y_class.shape[1]

    @property
    def regr_output_size(self):
        return self.y_regr.shape[1]

    def __getitem__(self, idx):
        out = {}
        out["x_ind"], out["x_data"] = get_row(self.x, idx)

        if self.class_output_size > 0:
            out["yc_ind"], out["yc_data"] = get_row(self.y_class, idx)

        if self.regr_output_size > 0:
            out["yr_ind"], out["yr_data"] = get_row(self.y_regr, idx)
            if self.y_censor.nnz > 0:
                out["ycen_ind"], out["ycen_data"] = get_row(self.y_censor, idx)

        return out

    def batch_to_x(self, batch, dev):
        """Takes 'xind' and 'x_data' from batch and converts them into a sparse tensor.
        Args:
            batch  batch
            dev    device to send the tensor to
        """
        return torch.sparse_coo_tensor(
                batch["x_ind"].to(dev),
                batch["x_data"].to(dev),
                size=[batch["batch_size"], self.x.shape[1]])

    def collate(self, batch):
        lists = {}
        for key in batch[0].keys():
            lists[key] = [b[key] for b in batch]

        out = {}
        out["x_ind"]  = to_idx_tensor(lists["x_ind"])
        out["x_data"] = torch.from_numpy(np.concatenate(lists["x_data"]))

        if "yc_ind" in lists:
            out["yc_ind"]  = to_idx_tensor(lists["yc_ind"])
            out["yc_data"] = torch.from_numpy(np.concatenate(lists["yc_data"]))

        if "yr_ind" in lists:
            out["yr_ind"]  = to_idx_tensor(lists["yr_ind"])
            out["yr_data"] = torch.from_numpy(np.concatenate(lists["yr_data"]))

        if "ycen_ind" in lists:
            out["ycen_ind"]  = to_idx_tensor(lists["ycen_ind"])
            out["ycen_data"] = torch.from_numpy(np.concatenate(lists["ycen_data"]))
        else:
            out["ycen_ind"]  = None
            out["ycen_data"] = None

        out["batch_size"] = len(batch)
        return out

