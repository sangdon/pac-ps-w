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
        

class PredSetConstructor_worst_rejection(PredSetConstructor):
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
        f_nll_list, w_lower_list, w_upper_list = [], [], []
        for x, y in ld:
            x, y = to_device(x, self.params.device), to_device(y, self.params.device)

            f_nll_i = self.mdl(x, y)
            w_itv_i, _ = self.mdl_iw(x, y, return_itv=True)
            f_nll_list.append(f_nll_i)
            w_lower_list.append(w_itv_i[0])
            w_upper_list.append(w_itv_i[1])
        f_nll_list, w_lower_list, w_upper_list = tc.cat(f_nll_list), tc.cat(w_lower_list), tc.cat(w_upper_list)
        assert(len(f_nll_list) == len(w_lower_list) == len(w_upper_list) == m)
        # f_nll_list, i_sort = f_nll_list.sort(ascending=False)
        # w_lower_list = i_lower_list[i_sort]
        # w_upper_list = i_upper_list[i_sort]

        ## iw max
        iw_max = self.mdl_iw.iw_max

        ## plot induced distribution and the corresponding IWs
        plot_induced_dist_iw(f_nll_list.cpu().detach().numpy(), w_lower_list.cpu().detach().numpy(), w_upper_list.cpu().detach().numpy(),
                             fn=os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'plot_induced_dist_iw'))


        ##----
        ## solve the minimax problem: minimize the prediction set size for the worst case iw
        ##----
        
        w_list = tc.zeros_like(f_nll_list)
        U = tc.rand(m, device=self.params.device) # sample only once
        
        ## find the smallest prediction set by line-searching over T
        T, T_step, T_end, T_opt_nll = 0.0, self.params.T_step, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -math.log(T) if T>0 else np.inf

            ## find the worst IW
            i_err = f_nll_list > T_nll
            w_list[i_err] = w_upper_list[i_err]
            w_list[~i_err] = w_lower_list[~i_err]

            ## run rejection sampling for target labeled examples
            i_accept = U <= (w_list/iw_max)
            f_nll_list_tar = f_nll_list[i_accept]

            # for u, w in zip(U, w_list):
            #     print(f'U = {u}, w = {w}, w/max = {w/iw_max}')
            
            ## CP bound
            error_U = (f_nll_list_tar > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.item(), len(f_nll_list_tar), delta
            #print(f'[Clopper-Pearson parametes] k={k_U}, n={n_U}, delta={delta_U}')
            U_bnd = bci_clopper_pearson_worst(k_U, n_U, delta_U)
            
            ## check condition
            if U_bnd <= eps:
                T_opt_nll = T_nll

            elif U_bnd >= self.params.eps_tol*eps: ## no more search if the upper bound is too large
                break

            print(f'[m = {m}, n = {n_U}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {math.exp(-T_opt_nll):.4f}, k = {k_U}, error_UCB = {U_bnd:.4f}')

            T += T_step


        ## save
        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.to(self.params.device)

        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
