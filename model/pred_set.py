import os, sys
import numpy as np

import torch as tc
import torch.nn as nn

from .util import *

##
## predictive confidence set
##
class PredSet(nn.Module):
    
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0):
        super().__init__()
        self.mdl = mdl
        self.T = nn.Parameter(tc.tensor(0.0))
        self.eps = nn.Parameter(tc.tensor(eps), requires_grad=False)
        self.delta = nn.Parameter(tc.tensor(delta), requires_grad=False)
        self.n = nn.Parameter(tc.tensor(n), requires_grad=False)

    
class PredSetCls(PredSet):
    """
    T \in [0, \infty]
    """
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0):
        super().__init__(mdl, eps, delta, n)

        
    def forward(self, x, y=None, e=0.0):
        with tc.no_grad():
            logp = self.mdl(x)['ph'].log()
            logp = logp + tc.rand_like(logp)*e # break the tie
            if y is not None:
                logp = logp.gather(1, y.view(-1, 1)).squeeze(1)
                        
        return -logp
        

    def set(self, x, y=None):
        with tc.no_grad():
            nlogp = self.forward(x)
            s = nlogp <= self.T
        return s

    
    def membership(self, x, y):
        with tc.no_grad():
            s = self.set(x, y)
            membership = s.gather(1, y.view(-1, 1)).squeeze(1)
        return membership

    
    def size(self, x, y=None):
        with tc.no_grad():
            sz = self.set(x, y).sum(1).float()
        return sz


class PredSetReg(PredSet):
    """
    T = - log(T'), where T' \in [0, \infty]
    T \in [-\infty, \infty]
    """
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, var_min=1e-16):
        super().__init__(mdl, eps, delta, n)
        self.var_min = var_min


    def forward(self, x, y=None, e=0.0):
        with tc.no_grad():
            assert(y is not None)
            logph = self.mdl(x, y)['logph']
            logph = logph + tc.rand_like(logph)*e # break the tie                        
        return -logph

    
    def set(self, x, y=None):
        """
        assumption: Gaussian with a diagonal covariance matrix
        return: an ellipsoid for each example
        """

        ## init
        T = self.T
        with tc.no_grad():
            ## label predictions
            out = self.mdl.forward(x)
        yh, yh_logvar = out['mu'], out['logvar']
        yh_var = tc.max(yh_logvar.exp(), tc.tensor(self.var_min, device=yh_logvar.device))
        yh, yh_logvar, yh_var = yh.reshape(yh.shape[0], -1), yh_logvar.reshape(yh.shape[0], -1), yh_var.reshape(yh.shape[0], -1)

        ## find the largest superlevel set of Gaussian at T
        d = yh.size(1)
        const = 2*T - d*np.log(2.0*np.pi) - yh_logvar.sum(1, keepdim=True)
        invalid = (const <= 0).squeeze() # when the superlevel set is empty
        axis_vector = yh_var.mul(const).sqrt() # imagine a high-dimensional ellipse with the axis vector (axis_vector)

        return yh, axis_vector, invalid
        

    def set_boxapprox(self, x):
        center, axis_vector, invalid = self.set(x)
        
        ## convert ellipses to boxes
        lb = center - axis_vector
        ub = center + axis_vector
        lb[invalid] = 0.0
        ub[invalid] = 0.0

        return lb, ub

    
    def membership(self, x, y):
        
        with tc.no_grad():
            css_membership = self.forward(x, y) <= self.T
        return css_membership

    
    def size_volume(self, x):
        _, axis_vector, invalid = self.set(x)
        size = axis_vector.prod(1) ## proportional to the volume of a high-dimensional ellipsoid
        size[invalid] = 0.0
        return size

    
    def size(self, x, size_type='volume'):
        if size_type == 'volume':
            return self.size_volume(x)
        else:
            raise NotImplementedError


            
