import os, sys
import numpy as np

import torch as tc
import torch.nn as nn
import torchvision.transforms.functional as TF

class Dummy(nn.Module):

    def forward(self, x, y=None):
        if y is not None:
            x['logph'] = x['logph_y'] ##TODO: better way?
        return x

class NoCal(nn.Module):
    def __init__(self, mdl, cal_target=-1):
        super().__init__()
        self.mdl = mdl
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
        
        ## return
        return {'yh_top': out['yh_top'],
                'yh_cal': out['yh_top'] if self.cal_target == -1 else tc.ones_like(out['yh_top'])*self.cal_target,
                'ph_cal': ph,
        }


def dist_mah(xs, cs, Ms, sqrt=True):
    diag = True if len(Ms.size()) == 2 else False
    assert(diag)
    assert(xs.size() == cs.size())
    assert(xs.size() == Ms.size())

    diff = xs - cs
    dist = diff.mul(Ms).mul(diff).sum(1)
    if sqrt:
        dist = dist.sqrt()
    return dist


def neg_log_prob(yhs, yhs_logvar, ys, var_min=1e-16):

    d = ys.size(1)
    yhs_var = tc.max(yhs_logvar.exp(), tc.tensor(var_min, device=yhs_logvar.device))
    loss_mah = 0.5 * dist_mah(ys, yhs, 1/yhs_var, sqrt=False)
    # if not all(loss_mah >= 0):
    #     print('loss_mah', loss_mah)
    #     print('ys', ys)
    #     print('yhs', yhs)
    #     print('yhs_var', yhs_var)
    #     print('yhs_logvar', yhs_logvar)
    assert(all(loss_mah >= 0))
    loss_const = 0.5 * np.log(2.0 * np.pi) * d
    loss_logdet = 0.5 * yhs_logvar.sum(1)
    loss = loss_mah + loss_logdet + loss_const

    return loss

# def g(grad):
#     print('grad_prev:', grad)
#     grad = -grad
#     print('grad_post:', grad)

#     return grad

class GradReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

        
    def forward(self, x, training=False):
        x = x * 1.0
        if training:
            if x.requires_grad:
                x.register_hook(lambda grad: -grad)
                #x.register_hook(g)
        return x

    
class ExampleNormalizer(nn.Module):
    def __init__(self, mdl, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): ## imagenet mean and std
        super().__init__()
        self.mdl = mdl
        self.mean = mean
        self.std = std

        
    def forward(self, x, **kwargs):
        x = TF.normalize(tensor=x, mean=self.mean, std=self.std)
        x = self.mdl(x, **kwargs)
        return x
    
