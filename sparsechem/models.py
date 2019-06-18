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
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        out = torch.mm(input, self.weight)
        if self.bias is not None:
            return out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.weight.shape[0], self.weight.shape[1], self.bias is not None
        )

class ModelConfig(object):
    def __init__(self,
        input_size,
        hidden_sizes,
        output_size,
        mapping            = None,
        hidden_dropout     = 0.0,
        last_dropout       = 0.0,
        weight_decay       = 0.0,
        input_size_freq    = None,
        tail_hidden_size   = None,
        non_linearity      = "relu",
        last_non_linearity = "relu",
    ):
        assert non_linearity in non_linearities.keys(), f"non_linearity can be either {non_linearities.keys()}."
        if mapping is not None:
            assert input_size == mapping.shape[0]

        self.mapping            = mapping
        self.input_size         = input_size
        self.hidden_sizes       = hidden_sizes
        self.output_size        = output_size
        self.hidden_dropout     = hidden_dropout
        self.last_dropout       = last_dropout
        self.last_non_linearity = last_non_linearity
        self.weight_decay       = weight_decay
        self.tail_hidden_size   = tail_hidden_size
        self.non_linearity      = non_linearity
        if input_size_freq is None:
            self.input_size_freq = input_size
        else:
            self.input_size_freq = input_size_freq
            assert self.input_size_freq <= input_size, f"Input size {input_size} is smaller than freq input size {self.input_size_freq}"

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

    def forward(self, X):
        if self.input_splits[1] == 0:
            return self.net_freq(X)
        Xfreq, Xrare = sparse_split2(X, self.input_splits[0], dim=1)
        return self.net_freq(Xfreq) + self.net_rare(Xrare)

class IntermediateNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(conf.hidden_sizes) - 1):
            self.intermediate_net.add_module(nn.Sequential(
                nn.ReLU(),
                nn.Dropout(conf.hidden_dropout),
                nn.Linear(conf.hidden_sizes[i], conf.hidden_sizes[i+1]),
            ))
    def forward(self, H):
        return self.net(H)

class LastNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        non_linearity = non_linearities[conf.last_non_linearity]
        self.net = nn.Sequential(
            non_linearity(),
            nn.Dropout(conf.last_dropout),
            nn.Linear(conf.hidden_sizes[-1], conf.output_size),
        )
    def forward(self, H):
        return self.net(H)

class SparseFFN(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.net = nn.Sequential(
            SparseInputNet(conf),
            IntermediateNet(conf),
            LastNet(conf),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) in [nn.Linear, SparseLinear]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, X):
        return self.net(X)


