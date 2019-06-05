import sparsechem as sc
import scipy.io
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

ecfp = scipy.io.mmread("chembl_23_x.mtx")
ic50 = scipy.io.mmread("chembl_23_y.mtx")

## changing ic50 to 0/1 encoding
ic50.data = (ic50.data == 1).astype(np.float32)

batch_size = 2000
dataset    = sc.SparseDataset(x=ecfp, y=ic50)
loader_tr  = DataLoader(dataset, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=sc.sparse_collate)

conf = sc.ModelConfig(
    input_size         = dataset.input_size,
    hidden_sizes       = [200],
    output_size        = dataset.output_size,
    #input_size_freq    = 1000,
    #tail_hidden_size   = 20,
    last_dropout       = 0.0,
    weight_decay       = 0.0,
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
        x_ind  = b["x_ind"].to(dev)
        x_data = b["x_data"].to(dev)
        y_ind  = b["y_ind"].to(dev)
        y_data = b["y_data"].to(dev)
        X      = torch.sparse_coo_tensor(x_ind, x_data, size=[b["batch_size"], dataset.input_size])

        yhat_all = net(X)
        yhat     = yhat_all[y_ind[0], y_ind[1]]
        
        output   = loss(yhat, y_data).mean()
        output.backward()

        optimizer.step()

        loss_sum   += output.detach()
        loss_count += 1

    loss_tr = loss_sum / loss_count
    print(f"Epoch {epoch}. loss_tr={loss_tr:.5f}")


