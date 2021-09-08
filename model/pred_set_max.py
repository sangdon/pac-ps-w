import os, sys
import numpy as np

import torch as tc
import torch.nn as nn
import torch.tensor as T

from .util import *
from . import PredSet


class PredSetMax(PredSet):
    """
    T \in [0, \infty]
    """
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0):
        super().__init__(mdl, eps, delta, n)

        
    def forward(self, x, y=None, e=1e-16):
        with tc.no_grad():
            v = self.mdl(x)
            v = v + tc.rand_like(v)*e # break the tie
        return v
        

    def set(self, x, y=None):
        raise NotImplementedError
        # with tc.no_grad():
        #     v = self.forward(x)
        #     s = v <= self.T
        # return s

    
    def membership(self, x, y=None):
        with tc.no_grad():
            v = self.forward(x)
            m = v <= self.T
        return m

    
    def size(self, x, y=None):
        sz = tc.ones(x.shape[0], device=x.device)*self.T
        return sz
