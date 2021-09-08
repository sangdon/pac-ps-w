import os, sys
import numpy as np
import pickle
import types
import itertools
import scipy
import math
import warnings

import torch as tc

from learning import *
from uncertainty import *
import model
from .util import *
    
    

class PredSetConstructor_worstiw(PredSetConstructor_CP):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

        
    def train(self, ld):

        m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        self.mdl.eps.data = self.mdl.eps.data / self.mdl_iw.iw_max
        super().train(ld)
        self.mdl.eps.data = tc.tensor(eps, device=self.mdl.eps.data.device)

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
