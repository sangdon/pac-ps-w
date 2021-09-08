import os, sys

import torch as tc
from torch import nn
import torch.nn.functional as F

class Temp(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl
        self.T = nn.Parameter(tc.tensor(1.0))
        
    
    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        x = self.mdl(x)['fh'] / self.T
        ##TODO: remove the redundancy due to the backward compatibility
        return {'fh': x, 'ph': F.softmax(x, -1), 'ph_cal': F.softmax(x, -1).max(-1)[0], 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0]}

    
    def train(self, train_flag=True):
        self.training = True
        self.mdl.eval()
        return self

    
    def eval(self):
        self.training = False
        self.mdl.eval()
        return self


    def parameters(self):
        return [self.T]
    

class TempReg(Temp):
    def __init__(self, mdl):
        super().__init__(mdl)

        
    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        out = self.mdl(x)
        ## consider exp(T) as temperature
        out['logvar'] = out['logvar'] - self.T.exp().log()
        return out
        
    
