import os, sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

import torch as tc
from torch.utils.data import TensorDataset, DataLoader

import data

class NormalDataset:

    def __init__(self, n_data, mu, sig, dim, slope):
        self.n_data = n_data
        self.mu = mu
        self.sig = sig
        assert(len(dim) == 1)
        self.dim = dim[0]
        self.slope = slope
        

    def __getitem__(self, index):
        x = np.concatenate((np.random.normal(self.mu, self.sig, 1).astype(np.single),
                            np.random.normal(self.mu, 1e-1, self.dim-1).astype(np.single))) # 1e-4
        p = 1.0 / (1.0 + np.exp(-x[0]*self.slope)) # first dimension determins class labels
        y = int(np.random.binomial(1, p))
        return x, y

    
    def __len__(self):
        return self.n_data

    
class Normal:
    def __init__(
            self, batch_size=100,
            sample_size={'train': 50000, 'val': 50000, 'test': 10000},
            dim=2048, 
            seed=0,
            num_workers=4,
            dist_params={'mu': 0.0, 'sig': 5.0, 'slope': 1.0},
            fn='plot_Normal',
            fontsize=15,
            **kwargs,
    ):
        
        ## init loaders
        ds = NormalDataset(sample_size['train'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ds = NormalDataset(sample_size['val'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ds = NormalDataset(sample_size['test'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)        
        
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')


class FatNormal(Normal):
    def __init__(
            self, batch_size=100,
            sample_size={'train': 100000, 'val': 100000, 'test': 10000},
            dim=2048,
            seed=0,
            num_workers=4,
            #dist_params={'mu': 0.0, 'sig': 5.0, 'slope': 5.0},
            dist_params={'mu': 0.0, 'sig': 5.0, 'slope': 5.0},
            fn='plot_FatNormal',
            fontsize=15,
            **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            sample_size=sample_size,
            dim=dim,
            seed=seed,
            num_workers=num_workers,
            dist_params=dist_params,
            fn=fn,
            fontsize=fontsize,
        )

        
class ThinNormal(Normal):
    def __init__(
            self, batch_size=100,
            sample_size={'train': 100000, 'val': 100000, 'test': 10000},
            dim=2048,
            seed=0,
            num_workers=4,
            #dist_params={'mu': 0.0, 'sig': 1.0, 'slope': 5.0},
            dist_params={'mu': 0.0, 'sig': 1.0, 'slope': 5.0},
            fn='plot_ThinNormal',
            fontsize=15,
            **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            sample_size=sample_size,
            dim=dim,
            seed=seed,
            num_workers=num_workers,
            dist_params=dist_params,
            fn=fn,
            fontsize=fontsize,
        )

class GoodThinNormal(Normal):
    def __init__(
            self, batch_size=100,
            sample_size={'train': 10000, 'val': 10000, 'test': 10000},
            dim=2048,
            seed=0,
            num_workers=4,
            dist_params={'mu': 0.0, 'sig': 1.0, 'slope': 1.0},
            fn='plot_GoodThinNormal',
            fontsize=15,
            **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            sample_size=sample_size,
            dim=dim,
            seed=seed,
            num_workers=num_workers,
            dist_params=dist_params,
            fn=fn,
            fontsize=fontsize,
        )

        
if __name__ == '__main__':
    dsld = data.Normal(100)
    print("#train = ", data.compute_num_exs(dsld.train))
    print("#val = ", data.compute_num_exs(dsld.val))
    print("#test = ", data.compute_num_exs(dsld.test))

"""
#train =  50000
#val =  10000
#test =  10000
"""
