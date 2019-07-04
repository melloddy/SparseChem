import numpy as np

def folding_rows(num_rows, train_size):
    if train_size > 1:
        num_tr = train_size
    else:
        num_tr = int(train_size * num_rows)
    rperm  = np.random.permutation(num_rows)
    idx_tr = rperm[:num_tr]
    idx_va = rperm[num_tr:]
    return idx_tr, idx_va
