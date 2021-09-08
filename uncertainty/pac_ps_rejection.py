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
    
    

class PredSetConstructor_rejection(PredSetConstructor):
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

        ## precompute -log f(y|x) and w(x)
        f_nll_list, w_list = [], []
        for x, y in ld:
            x, y = to_device(x, self.params.device), to_device(y, self.params.device)

            f_nll_i = self.mdl(x, y)
            w_i = self.mdl_iw(x, y)
            f_nll_list.append(f_nll_i)
            w_list.append(w_i)
        f_nll_list, w_list = tc.cat(f_nll_list), tc.cat(w_list)
        assert(len(f_nll_list) == len(w_list) == m)

        ## rejection sampling
        print(f'[iw_max] {self.mdl_iw.iw_max}')
        U = tc.rand(m, device=self.params.device)
        i_accept = U <= w_list/self.mdl_iw.iw_max
        f_nll_list = f_nll_list[i_accept]
    
        ## line search over T
        T, T_step, T_end, T_opt_nll = 0.0, self.params.T_step, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -math.log(T) if T>0 else np.inf

            ## CP bound
            error_U = (f_nll_list > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.item(), len(f_nll_list), delta
            print(f'[Clopper-Pearson parametes] k={k_U}, n={n_U}, delta={delta_U}')
            U = bci_clopper_pearson_worst(k_U, n_U, delta_U)
            
            ## check condition
            if U <= eps:
                T_opt_nll = T_nll

            elif U >= self.params.eps_tol*eps: ## no more search if the upper bound is too large
                break
            elif U >= 0.3:
                assert(eps < 0.3)
                break
                    
            print(f'[m = {m}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {math.exp(-T_opt_nll):.4f}, error_UCB = {U:.4f}')
            T += T_step        

        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.to(self.params.device)

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
