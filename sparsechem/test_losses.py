import sparsechem as sc
import torch
import numpy as np

def test_mse_censored_loss():
    tar = np.array([2.5, 2.0, 1.0, 0.5, -0.5, -1.0], dtype=np.float32)
    inp = np.array([1.5, 1.5, 1.5, 1.5,  1.5,  1.5], dtype=np.float32)
    cen1 = np.ones(6, dtype=np.float32)
    cen0 = np.zeros(6, dtype=np.float32)
    cen_n1 = -np.ones(6, dtype=np.float32)

    tar = np.concatenate([tar, tar, tar])
    inp = np.concatenate([inp, inp, inp])
    cen = np.concatenate([cen1, cen0, cen_n1])

    losses = sc.censored_mse_loss_numpy(target=tar, input=inp, censor=cen)
    exp = np.array([
        1.0, 0.25, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.25, 0.25, 1.0, 4.0, 6.25,
        0.0, 0.0,  0.25, 1.0, 4.0, 6.25,
    ], dtype=np.float32)
    assert np.allclose(losses, exp)


    tar = torch.FloatTensor([2.5, 2.0, 1.0, 0.5, -0.5, -1.0])
    inp = torch.FloatTensor([1.5, 1.5, 1.5, 1.5,  1.5,  1.5])
    cen1 = torch.ones(6)
    cen0 = torch.zeros(6)
    cen_n1 = -torch.ones(6)

    tar = torch.cat([tar, tar, tar])
    inp = torch.cat([inp, inp, inp])
    cen = torch.cat([cen1, cen0, cen_n1])

    losses = sc.censored_mse_loss(target=tar, input=inp, censor=cen)
    exp = torch.FloatTensor([
        1.0, 0.25, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.25, 0.25, 1.0, 4.0, 6.25,
        0.0, 0.0,  0.25, 1.0, 4.0, 6.25,
    ])
    assert np.allclose(losses.numpy(), exp.numpy())

def test_mae_censored_loss():
    tar = np.array([2.5, 2.0, 1.0, 0.5, -0.5, -1.0], dtype=np.float32)
    inp = np.array([1.5, 1.5, 1.5, 1.5,  1.5,  1.5], dtype=np.float32)
    cen1 = np.ones(6, dtype=np.float32)
    cen0 = np.zeros(6, dtype=np.float32)
    cen_n1 = -np.ones(6, dtype=np.float32)

    tar = np.concatenate([tar, tar, tar])
    inp = np.concatenate([inp, inp, inp])
    cen = np.concatenate([cen1, cen0, cen_n1])

    losses = sc.censored_mae_loss_numpy(target=tar, input=inp, censor=cen)
    exp = np.array([
        1.0, 0.5, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.5, 0.5, 1.0, 2.0, 2.5,
        0.0, 0.0, 0.5, 1.0, 2.0, 2.5,
    ], dtype=np.float32)
    assert np.allclose(losses, exp)

    tar = torch.FloatTensor([2.5, 2.0, 1.0, 0.5, -0.5, -1.0])
    inp = torch.FloatTensor([1.5, 1.5, 1.5, 1.5,  1.5,  1.5])
    cen1 = torch.ones(6)
    cen0 = torch.zeros(6)
    cen_n1 = -torch.ones(6)

    tar = torch.cat([tar, tar, tar])
    inp = torch.cat([inp, inp, inp])
    cen = torch.cat([cen1, cen0, cen_n1])

    losses = sc.censored_mae_loss(target=tar, input=inp, censor=cen)
    exp = torch.FloatTensor([
        1.0, 0.5, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.5, 0.5, 1.0, 2.0, 2.5,
        0.0, 0.0, 0.5, 1.0, 2.0, 2.5,
    ])
    assert np.allclose(losses.numpy(), exp.numpy())

