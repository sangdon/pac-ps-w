import os, sys

import torch as tc
from torch import nn

from .util import *
from model.util import neg_log_prob

def reduce(loss_vec, reduction):
    if reduction == 'mean':
        return loss_vec.mean()
    elif reduction == 'sum':
        return loss_vec.sum()
    elif reduction == 'none':
        return loss_vec
    else:
        raise NotImplementedError
    

##
## classification
##
def loss_xe(x, y, model, reduction='mean', device=tc.device('cpu')):
    x, y = x.to(device), y.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    loss = loss_fn(model(x)['fh'], y)
    return {'loss': loss}


def loss_xe_adv(x, y, model, reduction='mean', device=tc.device('cpu'), reg_param_adv=0.0):
    x, y, y_dom = x.to(device), y[0].to(device), y[1].to(device)
    loss_xe_fn = nn.CrossEntropyLoss(reduction=reduction)
    loss_adv_fn = nn.BCELoss(reduction=reduction)
    
    out = model(x)

    loss_xe = loss_xe_fn(out['fh'], y)
    loss_adv = loss_adv_fn(out['prob_src'][:, 0], y_dom.float())
    loss = loss_xe + reg_param_adv*loss_adv

    return {'loss': loss, 'loss_xe': loss_xe, 'loss_adv': loss_adv}


def loss_01(x, y, model, reduction='mean', device=tc.device('cpu')):
    x, y = x.to(device), y.to(device)
    yh = model(x)['yh_top']
    loss_vec = (yh != y).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}



##
## prediction set estimation
##
def loss_set_size(x, y, mdl, reduction='mean', device=tc.device('cpu')):
    x, y = to_device(x, device), to_device(y, device)
    loss_vec = mdl.size(x, y).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}


def loss_set_error(x, y, mdl, reduction='mean', device=tc.device('cpu')):
    x, y = to_device(x, device), to_device(y, device)
    loss_vec = (mdl.membership(x, y) == 0).float()
    loss = reduce(loss_vec, reduction)
    return {'loss': loss}
