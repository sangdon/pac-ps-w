import os, sys

import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class DANN(nn.Module):
    def __init__(self, mdl, mdl_adv):
        super().__init__()

        self.mdl = mdl
        self.mdl_adv = mdl_adv


    def forward(self, x, training=False):
        out = self.mdl(x, training=training)
        out_adv = self.mdl_adv(out['feat'], training=training)
        out['prob_src'] = out_adv['ph']
        out['fh_adv'] = out_adv['fh']
        
        return out
    
