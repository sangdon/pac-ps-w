import os, sys
from uncertainty import *
import numpy as np

import torch as tc


"""
estimate Split conformal preidction
"""
class SplitCPConstructor(PredSetConstructor):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix=name_postfix)
        
        if params:
            base = os.path.join(
                params.snapshot_root,
                params.exp_name,
                f"model_params{'_'+name_postfix if name_postfix else ''}_n_{self.mdl.n}_alpha_{self.mdl.eps:e}")
            self.mdl_fn_best = base + '_best'
            self.mdl_fn_final = base + '_final'
            self.mdl.to(self.params.device)

                
    def train(self, ld):
        n, alpha = self.mdl.n.item(), self.mdl.eps.item()
        print(f"## construct a prediction set via split conformal prediction: n = {n}, alpha = {alpha:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        ## read scores
        scores = []
        for x, y in ld:
            x, y = x.to(self.params.device), y.to(self.params.device)
            nlogp_i = self.mdl(x, y)
            scores.append(nlogp_i)
        scores.append(tc.tensor(np.inf, device=nlogp_i.device).unsqueeze(0))
        scores = tc.cat(scores)
        assert(n+1 == len(scores))
        #scores = scores[:n]
        
        scores_sorted = scores.sort(descending=False)[0]

        ## compute a quantile threshold
        k = int(np.ceil((n+1)*(1 - alpha)))
        q_opt = scores.kthvalue(k)[0]
        print(k, q_opt)
        self.mdl.T.data = q_opt

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        

class WeightedSplitCPConstructor(SplitCPConstructor):
    def __init__(self, model, mdl_iw=None, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix=name_postfix)
        self.mdl_iw = mdl_iw


    def train(self, ld):
        n, alpha = self.mdl.n.item(), self.mdl.eps.item()
        print(f"## construct a prediction set via weighted split conformal prediction: n = {n}, alpha = {alpha:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        ## read scores
        scores, ws = [], []
        for x, y in ld:
            x, y = x.to(self.params.device), y.to(self.params.device)
            nlogp_i = self.mdl(x, y)
            w = self.mdl_iw(x, y)
            scores.append(nlogp_i)
            ws.append(w)
        scores.append(tc.tensor([np.inf], device=nlogp_i.device))
        scores, ws = tc.cat(scores, 0), tc.cat(ws, 0)
        assert(n+1 == len(scores))
        
        ## save
        self.mdl.V_sorted.data, i_sorted = scores.sort()
        self.mdl.w_sorted.data = ws[i_sorted[:-1]]
        
        # ## save
        self._save_model(best=True)
        self._save_model(best=False)

        return True
        
