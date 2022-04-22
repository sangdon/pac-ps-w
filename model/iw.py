import os, sys
import numpy as np

import torch as tc
import torch.nn as nn

import model

class SourceDisc(nn.Module):
    def __init__(self, model_head, model_bb=None):
        super().__init__()
        self.model_head = model_head
        self.model_bb = model_bb
        #self.dim_feat = self.model_bb.dim_feat
        
        ## freeze backbone
        if self.model_bb is not None:
            for p in self.model_bb.parameters():
                p.requires_grad = False
                

    def forward(self, x, training=False):
        if self.model_bb is None:
            feat = x
        else:
            feat = self.model_bb(x, training=False)['feat']
        out = self.model_head(feat, training=training)
        return out
               

class IW(nn.Module):
    def __init__(self, mdl_sd, bound_type='mean', iw_max=1e9, iw_eps=1e-6):
        super().__init__()
        self.mdl_sd = mdl_sd
        self.bound_type = bound_type
        self.iw_max = tc.tensor(iw_max)
        self.iw_eps = iw_eps

        
    def forward(self, x, y=None, training=False, iw=True):
        out = self.mdl_sd(x, training=training)
        if 'ph_cal' in out:
            if self.bound_type == 'mean':
                g = out['ph_cal']
            elif self.bound_type == 'upper':
                g = out['ph_cal_lower']
            elif self.bound_type == 'lower':
                g = out['ph_cal_upper']
            else:
                raise NotImplementedError
        else:
            probs = out['ph']
            g = probs[:, 1]

        if iw:
            #w = tc.min(1/g - 1.0, self.iw_max.to(g.device))
            w = 1/g - 1.0
            
            w = w + tc.rand(w.shape[0], dtype=tc.float64, device=w.device)*self.iw_eps # break ties arbitrarily
            return w
        else:
            return g
        

class IWSDHist(nn.Module):
    def __init__(self, mdl_base, args_model, args_model_hist):
        super().__init__()

        self.mdl_sd = model.SourceDisc(getattr(model, args_model.sd)(args_model.feat_dim, 2), mdl_base)
        self.mdl_cal = getattr(model, args_model.sd_cal)(
            self.mdl_sd,
            delta=args_model_hist.delta,
            estimate_rate=args_model_hist.estimate_rate,
            cal_target=args_model_hist.cal_target)
        self.mdl_iw = model.IW(self.mdl_cal, bound_type='upper') ## choose the worst-case iw

        
    def forward(self, x):
        return self.mdl_iw(x)
    

class IWCal(nn.Module):
    def __init__(self, mdl_iw, n_bins, delta, iw_max=1e9):
        super().__init__()
        self.mdl_iw = mdl_iw
        self.n_bins = nn.Parameter(tc.tensor(n_bins), requires_grad=False)
        self.delta = nn.Parameter(tc.tensor(delta), requires_grad=False)
        self.bins = nn.Parameter(tc.zeros(n_bins+1), requires_grad=False)
        self.lower = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.upper = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.mean = nn.Parameter(tc.zeros(n_bins), requires_grad=False)
        self.iw_max = tc.tensor(iw_max)
        self.itv_sum = tc.tensor([0.0, 0.0])

        
    def forward(self, x, y=None, training=False, return_itv=False):
        with tc.no_grad():
            iw = self.mdl_iw(x, training=training)
        i = (iw.unsqueeze(1) >= self.bins[:-1].unsqueeze(0)) & (iw.unsqueeze(1) < self.bins[1:].unsqueeze(0))
        if not all(i.sum(1) ==1):
            print(i.sum(1))
            print(iw)
        assert(all(i.sum(1) ==1))
        i = i.int().argmax(1)
        l, u, m = self.lower[i], self.upper[i], self.mean[i]
        assert(all([l_i <= w_i < u_i for l_i, w_i, u_i in zip(self.bins[:-1][i], iw, self.bins[1:][i])]))
        
        if return_itv:
            return (l, u), i
        else:
            return m

