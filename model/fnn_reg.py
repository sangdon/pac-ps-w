import os, sys

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class FNNReg(nn.Module):
    def __init__(self, n_in, n_out, n_hiddens=500, n_layers=4):
        super().__init__()
        
        model_feat = []
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            models.append(nn.Linear(n, n_hiddens))
            models.append(nn.ReLU())
            models.append(nn.Dropout(0.5))
        self.model_feat = nn.Sequential(*model_feat)
        self.model_mu = nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out)
        self.model_logvar = nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out)
        self.models = [self.model_feat, self.model_mu, self.model_logvar]
        
        
    def forward(self, x, training=False):
        if training:
            [m.train() for m in self.models]
        else:
            [m.eval() for m in self.models]

        z = self.model_feat(x)
        mu, logvar = self.model_mu(z), self.model_logvar(z)
        
        return {'mu': mu, 'logvar': logvar}


class LinearReg(FNNReg):
    def __init__(self, n_in, n_out, n_hiddens=None):
        super().__init__(n_in, n_out, n_hiddens=None, n_layers=0)


class SmallFNNReg(FNNReg):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1)

    
class MidFNNReg(FNNReg):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2)

        
class BigFNNReg(FNNReg):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4)






