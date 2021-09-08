import os, sys
import warnings

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from .fnn import FNN
from .util import GradReversalLayer


class AdvFNN(FNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def forward(self, x, training=False):
        x = GradReversalLayer()(x, training=training)
        return super().forward(x, training)
    

class AdvLinear(AdvFNN):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, n_layers=0)


class SmallAdvFNN(AdvFNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1)

    
class MidAdvFNN(AdvFNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2)

        
class BigAdvFNN(AdvFNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4)





