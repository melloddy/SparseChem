# Copyright (c) 2020 KU Leuven
import torch
import math
from torch import nn

non_linearities = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

class SparseLinear(torch.nn.Module):
    """
    Linear layer with sparse input tensor, and dense output.
        in_features    size of input
        out_features   size of output
        bias           whether to add bias
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) / math.sqrt(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        out = torch.mm(input, self.weight)
        if self.bias is not None:
            return out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.weight.shape[0], self.weight.shape[1], self.bias is not None
        )

def sparse_split2(tensor, split_size, dim=0):
    """
    Splits tensor into two parts.
    Args:
        split_size   index where to split
        dim          dimension which to split
    """
    assert tensor.layout == torch.sparse_coo
    indices = tensor._indices()
    values  = tensor._values()

    shape  = tensor.shape
    shape0 = shape[:dim] + (split_size,) + shape[dim+1:]
    shape1 = shape[:dim] + (shape[dim] - split_size,) + shape[dim+1:]

    mask0 = indices[dim] < split_size
    X0 = torch.sparse_coo_tensor(
            indices = indices[:, mask0],
            values  = values[mask0],
            size    = shape0)

    indices1       = indices[:, ~mask0]
    indices1[dim] -= split_size
    X1 = torch.sparse_coo_tensor(
            indices = indices1,
            values  = values[~mask0],
            size    = shape1)
    return X0, X1



class SparseInputNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        if conf.input_size_freq is None:
            conf.input_size_freq = conf.input_size
        assert conf.input_size_freq <= conf.input_size, f"Number of high important features ({conf.input_size_freq}) should not be higher input size ({conf.input_size})."
        self.input_splits = [conf.input_size_freq, conf.input_size - conf.input_size_freq]
        self.net_freq   = SparseLinear(self.input_splits[0], conf.hidden_sizes[0])

        if self.input_splits[1] == 0:
            self.net_rare = None
        else:
            ## TODO: try adding nn.ReLU() after SparseLinear
            ## Bias is not needed as net_freq provides it
            self.net_rare = nn.Sequential(
                SparseLinear(self.input_splits[1], conf.tail_hidden_size),
                nn.Linear(conf.tail_hidden_size, conf.hidden_sizes[0], bias=False),
            )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == SparseLinear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            m.bias.data.fill_(0.1)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)


    def forward(self, X):
        if self.input_splits[1] == 0:
            return self.net_freq(X)
        Xfreq, Xrare = sparse_split2(X, self.input_splits[0], dim=1)
        return self.net_freq(Xfreq) + self.net_rare(Xrare)

class MiddleNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(conf.hidden_sizes) - 1):
            self.net.add_module(f"layer_{i}", nn.Sequential(
                nn.ReLU(),
                nn.Dropout(conf.middle_dropout),
                nn.Linear(conf.hidden_sizes[i], conf.hidden_sizes[i+1], bias=True),
            ))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, H):
        return self.net(H)

class LastNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.non_linearity = conf.last_non_linearity
        non_linearity = non_linearities[conf.last_non_linearity]
        self.net = nn.Sequential(
            non_linearity(),
            nn.Dropout(conf.last_dropout),
            nn.Linear(conf.hidden_sizes[-1], conf.output_size),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("sigmoid"))
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def forward(self, H):
        return self.net(H)

class SparseFFN(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        if hasattr(conf, "class_output_size"):
            self.class_output_size = conf.class_output_size
            self.regr_output_size  = conf.regr_output_size
        else:
            self.class_output_size = None
            self.regr_output_size  = None

        self.net = nn.Sequential(
            SparseInputNet(conf),
            MiddleNet(conf),
            LastNet(conf),
        )

    @property
    def has_2heads(self):
        return self.class_output_size is not None

    def forward(self, X, last_hidden=False):
        if last_hidden:
            H = self.net[:-1](X)
            return self.net[-1].net[:-1](H)
        out = self.net(X)
        if self.class_output_size is None:
            return out
        ## splitting to class and regression
        if self.class_output_size == 0: return None, out
        if self.regr_output_size == 0:  return out, None
        return out[:, :self.class_output_size], out[:, self.class_output_size:]

