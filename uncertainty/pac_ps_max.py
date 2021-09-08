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

    

class PredSetMaxConstructor(PredSetConstructor):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

        
    def train(self, ld):
        n, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        assert(self.params.method == 'pac_predset_CP')

        ## precompute values
        v_list = []
        for x, _ in ld:
            x = to_device(x, self.params.device)

            v_i = self.mdl(x)
            v_list.append(v_i)
        v_list = tc.cat(v_list)
        assert(len(v_list) == n)

        ## line search over T
        T, T_step, T_end, T_opt, k_U_opt = 0.0, self.params.T_step, self.params.T_end, np.inf, 0
        while T <= T_end:
            ## CP bound
            error_U = (v_list > T).sum().float()
            k_U, n_U, delta_U = error_U.int().item(), n, delta
            #print(f'[Clopper-Pearson parametes] k={k_U}, n={n_U}, delta={delta_U}')
            U = bci_clopper_pearson_worst(k_U, n_U, delta_U)

            print(f'[n = {n}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {T_opt:.4f}, #error = {k_U}, error_emp = {k_U/n_U:.4f}, U = {U:.6f}')

            if U <= eps:
                k_U_opt = k_U
                T_opt = T
                break
            T += T_step        

        self.mdl.T.data = tc.tensor(T_opt)
        self.mdl.to(self.params.device)

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
