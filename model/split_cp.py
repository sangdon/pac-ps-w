import os, sys
import numpy as np

import torch as tc
import torch.nn as nn
import torch.tensor as T

from .util import *
from .pred_set import *

##
## predictive confidence set
##
SplitCPCls = PredSetCls
SplitCPReg = PredSetReg


class WeightedSplitCPCls(SplitCPCls):
    def __init__(self, mdl, mdl_iw, eps, delta, n):
        super().__init__(mdl, eps, delta, n)
        self.alpha = self.eps
        self.mdl_iw = mdl_iw
        self.V_sorted = nn.Parameter(tc.zeros(n+1))
        self.w_sorted = nn.Parameter(tc.zeros(n))

        
    def set(self, x, y=None):
        assert(len(self.V_sorted) == self.n+1)
        assert(len(self.w_sorted) == self.n)
        bs = x.shape[0]
        
        ## compute threshold (i.e., 1-alpha quantile) and construct a prediction set
        with tc.no_grad():
            V_new = self.forward(x)
            w_new = self.mdl_iw(x, y)

        w_sorted = self.w_sorted.expand([bs, -1])
        w_sorted = tc.cat([w_sorted, w_new.unsqueeze(1)], 1)
        p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
        p_sorted_acc = p_sorted.cumsum(1)
        
        i_T = tc.argmax((p_sorted_acc >= 1.0 - self.alpha).int(), dim=1, keepdim=True)
        T = self.V_sorted.expand([bs, -1]).gather(1, i_T)
        s = V_new <= T
            
        return s

        
        

        
