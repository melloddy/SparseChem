# Copyright (c) 2020 KU Leuven
import torch
import math
import numpy as np
from torch import nn

non_linearities = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

class Scaling(torch.nn.Module):
    """
    Elementwise scaling layer
    features    size
    bias        wether to add bias
    """
    def __init__(self, features, bias=True):
        super(Scaling, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        out = input * self.weight 
        if self.bias is not None:
            return out + self.bias
        return out

    def GetRegularizer(self):
        return torch.norm((self.weight - 1))

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
                non_linearities[conf.middle_non_linearity](),
                nn.Dropout(conf.dropouts_trunk[i]),
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
    def __init__(self, conf, extra_input_size=0, output_size = None, last_non_linearity = None, last_hidden_sizes = None, dropouts = None):
        super().__init__()

        if last_non_linearity is None:
            last_non_linearity = conf.last_non_linearity
        if output_size is None:
            output_size = conf.output_size

        if last_hidden_sizes is None:
            if conf.last_hidden_sizes is None:
                last_hidden_sizes = []
            else:
                last_hidden_sizes = conf.last_hidden_sizes

        
        self.net = nn.Sequential()
        if len(last_hidden_sizes) > 0:
            output_size_initial = last_hidden_sizes[0]
        else:
            output_size_initial = output_size

        self.net.add_module(f"initial_layer", nn.Sequential(
            non_linearities[last_non_linearity](),
            nn.Dropout(conf.dropouts_trunk[-1]),
            nn.Linear(conf.hidden_sizes[-1]+extra_input_size, output_size_initial),
        ))
        for i in range(len(last_hidden_sizes) - 1):
            self.net.add_module(f"layer_{i}", nn.Sequential(
                non_linearities[last_non_linearity](),
                nn.Dropout(dropouts[i]),
                nn.Linear(last_hidden_sizes[i], last_hidden_sizes[i+1], bias=True),
            ))
        if len(last_hidden_sizes) > 0:
            self.net.add_module(f"last_layer", nn.Sequential(
                 non_linearities[last_non_linearity](),
                 nn.Dropout(dropouts[-1]),
                 nn.Linear(last_hidden_sizes[-1], output_size),
            ))

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
       #     LastNet(conf), #made it separate
        )

        self.scaling = None #Scaling(conf.hidden_sizes[-1])
        if self.class_output_size is None or self.regr_output_size is None:
            raise ValueError("Both regression and classification tergets are needed for hybrid mode")
        self.classLast = LastNet(conf, output_size = conf.class_output_size, last_non_linearity = 'relu', last_hidden_sizes = conf.last_hidden_sizes_class, dropouts = conf.dropouts_class) #Override output size
        if conf.scaling_regularizer == np.inf:
            self.regrLast  =  nn.Sequential(
                    LastNet(conf, output_size = conf.regr_output_size, last_non_linearity = 'tanh', last_hidden_sizes = conf.last_hidden_sizes_reg, dropouts = conf.dropouts_reg),
                    )
        else:
            self.scaling = Scaling(conf.hidden_sizes[-1])
            self.regrLast  =  nn.Sequential(
                    self.scaling,
                    LastNet(conf, output_size = conf.regr_output_size, last_non_linearity = 'tanh', last_hidden_sizes = conf.last_hidden_sizes_reg, dropouts = conf.dropouts_reg),
                   )
            self.scaling_regularizer = conf.scaling_regularizer
        
        regmask = torch.ones(1,conf.hidden_sizes[-1])
        regmask[:,:-conf.regression_feature_size] = 0.0
        classmask = torch.ones(1,conf.hidden_sizes[-1])
        classmask[:,conf.class_feature_size:] = 0.0
        self.register_buffer('regmask', regmask)
        self.register_buffer('classmask', classmask)


    def GetRegularizer(self):
        if self.scaling is not None:
            return self.scaling_regularizer * self.scaling.GetRegularizer()
        else:
            return 0;

    @property
    def has_2heads(self):
        return self.class_output_size is not None

    def forward(self, X, last_hidden=False):
        if last_hidden:
            raise ValueError("Last_hidden is not supported for hybrid mode")
  #          H = self.net[:-1](X)
  #          return self.net[-1].net[:-1](H)
        out = self.net(X)
  #      if self.class_output_size is None:
  #          return out
        ## splitting to class and regression

        return self.classLast(out * self.classmask), self.regrLast(out * self.regmask)

class SparseFFN_combined(nn.Module):
  def __init__(self, conf, shared_trunk, local_trunk, head):
    super().__init__()
    if hasattr(conf, "class_output_size"):
       self.class_output_size = conf.class_output_size
       self.regr_output_size  = conf.regr_output_size
    else:
       self.class_output_size = None
       self.regr_output_size  = None
    self.shared_trunk = shared_trunk
    self.local_trunk  = local_trunk
    self.head = head

  @property
  def has_2heads(self):
    return self.class_output_size is not None
  def forward(self, input):
    shared_output = self.shared_trunk(input)
    if self.local_trunk is not None:
        local_output  = self.local_trunk(input)
        combined = torch.cat((shared_output,local_output), dim=1)
    else:
        combined = shared_output

    out = self.head(combined)
    if self.class_output_size is None:
       return out
     ## splitting to class and regression
    return out[:, :self.class_output_size], out[:, self.class_output_size:]

def censored_mse_loss(input, target, censor, censored_enabled=True):
    """
    Computes for each value the censored MSE loss.
    Args:
        input     tensor of predicted values
        target    tensor of true values
        censor    tensor of censor masks: -1 lower, 0 no and +1 upper censoring.
    """
    y_diff = target - input
    if censor is not None and censored_enabled:
        y_diff = torch.where(censor==0, y_diff, torch.relu(censor * y_diff))
    return y_diff * y_diff

def censored_mae_loss(input, target, censor):
    """
    Computes for each value the censored MAE loss.
    Args:
        input    tensor of predicted values
        target   tensor of true values
        censor   tensor of censor masks: -1 lower, 0 no and +1 upper censoring.
    """
    y_diff = target - input
    if censor is not None:
        y_diff = torch.where(censor==0, y_diff, torch.relu(censor * y_diff))
    return torch.abs(y_diff)

def censored_mse_loss_numpy(input, target, censor):
    """
    Computes for each value the censored MSE loss in *Numpy*.
    Args:
        input     tensor of predicted values
        target    tensor of true values
        censor    tensor of censor masks: -1 lower, 0 no and +1 upper censoring.
    """
    y_diff = target - input
    if censor is not None:
        y_diff = np.where(censor==0, y_diff, np.clip(censor * y_diff, a_min=0, a_max=None))
    return y_diff * y_diff

def censored_mae_loss_numpy(input, target, censor):
    """
    Computes for each value the censored MSE loss in *Numpy*.
    Args:
        input     tensor of predicted values
        target    tensor of true values
        censor    tensor of censor masks: -1 lower, 0 no and +1 upper censoring.
    """
    y_diff = target - input
    if censor is not None:
        y_diff = np.where(censor==0, y_diff, np.clip(censor * y_diff, a_min=0, a_max=None))
    return np.abs(y_diff)
