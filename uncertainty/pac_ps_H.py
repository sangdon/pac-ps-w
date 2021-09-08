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
        

class PredSetConstructor_H(PredSetConstructor):
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
            w_i = self.mdl_iw(x)
            f_nll_list.append(f_nll_i)
            w_list.append(w_i)
        f_nll_list, w_list = tc.cat(f_nll_list), tc.cat(w_list)
        assert(len(f_nll_list) == len(w_list) == m)
        
        ## line search over T
        T, T_step, T_end, T_opt_nll = 0.0, self.params.T_step, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -math.log(T) if T>0 else np.inf

            ## Hoeffdings bound
            mean_emp_H = ((f_nll_list > T_nll).float()*w_list).mean().item()
            n_H, a_H, b_H, delta_H = m, 0.0, self.params.iw_max, delta
            #print(f'[hoeffding parametes] mean_emp={mean_emp_H}, n={n_H}, a={a_H}, b={b_H}, delta={delta_H}')
            H = estimate_mean_worst_hoeffding(mean_emp_H, n_H, a_H, b_H, delta_H)
            H = min(H, self.params.iw_max)

            if H <= eps:
                T_opt_nll = T_nll
            elif H >= self.params.eps_tol*eps: ## no more search if the upper bound is too large
                break
            print(f'[m = {m}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {math.exp(-T_opt_nll):.4f}, error_emp = {mean_emp_H:.4f}, H = {H:.4f}')
            T += T_step        

        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.to(self.params.device)

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
