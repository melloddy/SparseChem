from torch.utils.data import Dataset
import torch
import scipy.sparse
import numpy as np

class SparseDataset(Dataset):
    def __init__(self, x, y):
        '''
        Args:
            X (sparse matrix): input [n_sampes, features_in]
            Y (sparse matrix): output [n_samples, features_out]
        '''
        self.x = x.tocsr(copy=False).astype(np.float32)
        self.y = y.tocsr(copy=False)

        assert self.x.shape[0]==self.y.shape[0], f"Input has {self.x.shape[0]} rows, output has {self.y.shape[0]} rows."

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        xi = self.x[idx,:]
        yi = self.y[idx,:]

        return {
            "x_ind":  xi.indices,
            "x_data": xi.data,
            "y_ind":  yi.indices,
            "y_data": yi.data,
        }

class MappingDataset(Dataset):
    def __init__(self, x_ind, x_data, y, mapping=None):
        """
        Dataset that creates a mapping for features of x (0...N_feat-1).
        """
        pass

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
    y_row  = np.repeat(np.arange(len(y_ind)), [len(i) for i in y_ind])
    y_col  = np.concatenate(y_ind).astype(np.int64)

    return {
        "x_ind":  torch.LongTensor([xrow, xcol]),
        "x_data": torch.from_numpy(xv),
        "y_row":  torch.from_numpy(y_row),
        "y_col":  torch.from_numpy(y_col),
        "y_data": torch.from_numpy(np.concatenate(y_data)),
        "batch_size": len(batch),
    }

