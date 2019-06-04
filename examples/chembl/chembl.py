import sparsechem as sc
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader

ecfp = scipy.io.mmread("chembl_23_x.mtx")
ic50 = scipy.io.mmread("chembl_23_y.mtx")

batch_size = 10
dataset    = sc.SparseDataset(x=ecfp, y=ic50)
loader_tr  = DataLoader(dataset, batch_size=batch_size, num_workers = 0, pin_memory=True, collate_fn=sc.sparse_collate)

conf = sc.ModelConfig(
    input_size   = dataset.input_size,
    hidden_sizes = [100],
    output_size  = dataset.output_size,
    last_dropout = 0.1,
    weight_decay = 1e-3,
    last_non_linearity = "relu",
)

dev = "cuda:0"
net = sc.SparseFFN(conf).to(dev)

for b in loader_tr:
    x_ind  = b["x_ind"].to(dev)
    x_data = b["x_data"].to(dev)
    out    = net(x_ind=x_ind, x_data=x_data, num_rows=b["batch_size"])
    break

