import os, sys
import time
import warnings
import pickle

from .util import *
#from learning import *
#from uncertainty import *

# import learning
# import uncertainty
# import model

import torch as tc

class IWTwoNormals(tc.nn.Module):
    def __init__(self, p_params, q_params, device):
        super().__init__()
        dim = p_params['mu'].shape[0]
        self.p_normal = tc.distributions.LowRankMultivariateNormal(p_params['mu'].to(device),
                                                                   tc.zeros(dim, dim, device=device),
                                                                   p_params['sig'].to(device)**2)
        self.q_normal = tc.distributions.LowRankMultivariateNormal(q_params['mu'].to(device),
                                                                   tc.zeros(dim, dim, device=device),
                                                                   q_params['sig'].to(device)**2)

        ## max iw
        assert(all(p_params['mu'] == q_params['mu']))
        x = p_params['mu'].to(device)
        p_logprob = self.p_normal.log_prob(x)
        q_logprob = self.q_normal.log_prob(x)

        iw_log = q_logprob - p_logprob
        self.iw_max = iw_log.squeeze().exp()

        
    def forward(self, x, y=None, training=False):
        if training:
            self.train()
        else:
            self.eval()

        p_logprob = self.p_normal.log_prob(x)
        q_logprob = self.q_normal.log_prob(x)

        iw_log = q_logprob - p_logprob
        return iw_log.squeeze().exp()


    
def get_two_gaussian_true_iw(args):
    ## load true IW
    assert('Normal' in args.data.src and 'Normal' in args.data.tar)
    if args.data.src == 'FatNormal':
        mu = tc.zeros(args.data.dim)
        sig = tc.ones(args.data.dim) * 1e-1
        sig[0] = 5.0
        p_params = {'mu': mu, 'sig': sig}
    elif args.data.src == 'ThinNormal':
        mu = tc.zeros(args.data.dim)
        sig = tc.ones(args.data.dim) * 1e-1
        sig[0] = 1.0
        p_params = {'mu': mu, 'sig': sig}
    else:
        raise NotImplementedError

    if args.data.tar == 'FatNormal':
        mu = tc.zeros(args.data.dim)
        sig = tc.ones(args.data.dim) * 1e-1
        sig[0] = 5.0
        q_params = {'mu': mu, 'sig': sig}
    elif args.data.tar == 'ThinNormal':
        mu = tc.zeros(args.data.dim)
        sig = tc.ones(args.data.dim) * 1e-1 
        sig[0] = 1.0
        q_params = {'mu': mu, 'sig': sig}
    else:
        raise NotImplementedError
        
    mdl_iw = IWTwoNormals(p_params=p_params, q_params=q_params, device=args.device)

        
