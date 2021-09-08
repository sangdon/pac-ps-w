import os, sys
import time
#import numpy as np

#import torch as tc
#import torch.tensor as T
#import torch.nn as nn

# sys.path.append("../")
# from .calibrator import BaseCalibrator
# from classification.utils import *

from learning import *
from uncertainty import *


class ClsCalibrator(ClsLearner):
    
    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        
        ## compute classification error
        error, ece, *_ = super().test(ld, mdl, loss_fn)

        ## compute confidence distributions
        ph_list = compute_conf(self.mdl if mdl is None else mdl, ld, self.params.device)
        mn = ph_list.min()
        Q1 = ph_list.kthvalue(int(round(ph_list.size(0)*0.25)))[0]
        Q2 = ph_list.median()
        Q3 = ph_list.kthvalue(int(round(ph_list.size(0)*0.75)))[0]
        mx = ph_list.max()
        av = ph_list.mean()
        ph_dist = {'min': mn, 'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'max': mx, 'mean': av}

        if verbose:
            print('[test%s, %f secs.] test error = %.2f%%, ece = %.2f%%'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0, ece*100.0))
            print(
                f'[ph distribution] '
                f'min = {mn:.4f}, 1st-Q = {Q1:.4f}, median = {Q2:.4f}, 3rd-Q = {Q3:.4f}, max = {mx:.4f}, mean = {av:.4f}'
            )

        return error, ece, ph_list

    

class TempScalingLearner(ClsCalibrator):
    def __init__(self, mdl, params=None, name_postfix='cal_temp'):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_xe
        self.loss_fn_test = loss_01
        self.T_min = 1e-9
        
        
    def _train_epoch_batch_end(self, i_epoch):
        [T.data.clamp_(self.T_min) for T in self.mdl.parameters()]




class HistBinLearner(ClsCalibrator):
    
    def __init__(self, mdl, params=None, name_postfix='cal_hist'):
        super().__init__(mdl, params, name_postfix)

        
    def _learn_histbin(self, ph_list, c_list):
        n_val = c_list.shape[0]
        ph_list = ph_list.clone().detach()
        c_list = c_list.clone().detach()
        n_bins = self.mdl.n_bins

        ## compute histogram bin
        ch, lower, upper, n_exs, lower_rate, upper_rate = [], [], [], [], [], []
        for l, u in zip(self.mdl.bins[:-1], self.mdl.bins[1:]):
            idx = (ph_list>=l) & (ph_list<=u)
            c_list_i = c_list[idx]
            n = idx.sum()
            x = c_list_i.float().sum()
            if n>=1:
                if self.mdl.estimate_rate:
                    delta = self.mdl.delta/n_bins/2.0
                else:
                    delta = self.mdl.delta/n_bins
                    
                # confidence interval of confidence
                ch_l_ci, ch_u_ci = bci_clopper_pearson(x.item(), n.item(), delta.item())
                ch_mean = c_list_i.float().mean().item()
                # confidence interval of rate
                if self.mdl.estimate_rate:
                    l_ci_rate, u_ci_rate = bci_clopper_pearson(n.item(), n_val, delta.item())
            else:
                ch_l_ci, ch_u_ci = 0.0, 1.0
                ch_mean = 0.5
                l_ci_rate, u_ci_rate = 0.0, 1.0

            ch.append(ch_mean)
            lower.append(ch_l_ci)
            upper.append(ch_u_ci)
            n_exs.append(n)
            lower_rate.append(l_ci_rate)
            upper_rate.append(u_ci_rate)

        ## save
        self.mdl.ch.data = tc.tensor(ch)
        self.mdl.lower.data = tc.tensor(lower)
        self.mdl.upper.data = tc.tensor(upper)
        self.mdl.n_exs.data = tc.tensor(n_exs)
        self.mdl.n_val.data = tc.tensor(n_val)
        if self.mdl.estimate_rate:
            self.mdl.lower_rate.data = tc.tensor(lower_rate)
            self.mdl.upper_rate.data = tc.tensor(upper_rate)
            self.mdl.est_rate.data = self.mdl.n_exs.data.float() / self.mdl.n_val.data.float()



    def train(self, ld_tr, ld_val, ld_test=None):
        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return
        
        ## load confidence prediction and correctness on label prediction
        self.mdl.eval()
        ph_list, c_list = [], []
        for x, y in ld_val:
            x, y = x.to(self.params.device), y.to(self.params.device)

            with tc.no_grad():
                ph = self.mdl.mdl(x)['ph']
                if self.mdl.cal_target == -1:
                    ph, yh = ph.max(1)
                elif self.mdl.cal_target in range(ph.shape[1]):
                    ph, yh = ph[:, self.mdl.cal_target], tc.ones(ph.shape[0], device=ph.device).long()*self.mdl.cal_target
                else:
                    raise NotImplementedError
                c = (yh == y).float()
            ph_list.append(ph)
            c_list.append(c)
        ph_list, c_list = tc.cat(ph_list), tc.cat(c_list)

        ## learn PAC histogram
        self._learn_histbin(ph_list, c_list)
        self._save_model()

        ## save the model
        self._train_end(ld_val, ld_test)

        ## summary results
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'plot_histbin%s'%('_'+self.name_postfix if self.name_postfix else '')) 
        plot_histbin(self.mdl.bins.cpu().numpy(), self.mdl.ch.cpu().numpy(), self.mdl.lower.cpu().numpy(), self.mdl.upper.cpu().numpy(),
                     self.mdl.n_exs.cpu().numpy(), fn)
        if self.mdl.estimate_rate:
            fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'plot_histbin_rate%s'%('_'+self.name_postfix if self.name_postfix else '')) 
            plot_histbin(self.mdl.bins.cpu().numpy(), self.mdl.est_rate.cpu().numpy(), self.mdl.lower_rate.cpu().numpy(), self.mdl.upper_rate.cpu().numpy(),
                         self.mdl.n_exs.cpu().numpy(),fn)
        
        
        
