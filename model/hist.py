import os, sys

import torch as tc
import torch.nn as nn


class HistBin(nn.Module):
    def __init__(self, mdl, delta, n_bins=20, estimate_rate=False, cal_target=-1):
        super().__init__()

        self.mdl = mdl
        self.delta = nn.Parameter(tc.tensor(delta), requires_grad=False)
        # construct static bins
        self.n_bins = nn.Parameter(tc.tensor(n_bins), requires_grad=False)
        self.bins = nn.Parameter(tc.linspace(0.0, 1.0, n_bins+1), requires_grad=False)
        self.ch = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.lower = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.upper = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.n_exs = nn.Parameter(tc.zeros(n_bins).long(), requires_grad=False)
        self.n_val = nn.Parameter(tc.tensor(0), requires_grad=False)
        self.estimate_rate = nn.Parameter(tc.tensor(estimate_rate).long(), requires_grad=False)##TODO: do I need this?
        if estimate_rate:
            self.lower_rate = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
            self.upper_rate = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
            self.est_rate = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.cal_target = nn.Parameter(tc.tensor(cal_target).long(), requires_grad=False)
            

    def forward(self, x, training=False):
        assert(training==False)
        self.eval() ##always

        ## forward along the base model
        out = self.mdl(x)
        if self.cal_target == -1:
            ph = out['ph_top']
        elif self.cal_target in range(out['ph'].shape[1]):
            ph = out['ph'][:, self.cal_target]
        else:
            raise NotImplementedError
        cal_target_str = str(self.cal_target)
        
        ## calibrate
        i = ((ph.unsqueeze(1) >= self.bins[:-1].unsqueeze(0)) & (ph.unsqueeze(1) <= self.bins[1:].unsqueeze(0))).int().argmax(1)
        l, u, m = self.lower[i], self.upper[i], self.ch[i]

        ## return
        return {'yh_top': out['yh_top'],
                'yh_cal': out['yh_top'] if self.cal_target == -1 else tc.ones_like(out['yh_top'])*self.cal_target,
                'ph_cal_lower': l,
                'ph_cal_upper': u,
                'ph_cal': m,
        }
    
    
                
                
