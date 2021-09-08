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

        # ## draw examples
        # np.random.seed(seed)
        # n = sample_size['train'] + sample_size['val'] + sample_size['test']
        # if dim==1:
        #     x = np.random.normal(dist_params['mu'], dist_params['sig'], (n,1)).astype(np.float)
        # else:
        #     assert(dim > 1)
        #     x = np.random.multivariate_normal(
        #         [dist_params['mu']]*dim,
        #         np.diag([dist_params['sig']]*dim),
        #         (n,)).astype(np.float)
        #     w = [1.0] + [0.0]*(dim-1)
        # p = 1.0 / (1.0 + np.exp(-x[:, 0]*dist_params['slope'])) # first dimension determins class labels
        # y = np.random.binomial(1, p).astype(np.int)
        # np.random.seed(int(time.time()))
        
        # ## split
        # a, b, c = sample_size['train'], sample_size['val'], sample_size['test']
        # x_train, x_val, x_test = x[:a, :], x[a:a+b, :], x[a+b:, :]
        # y_train, y_val, y_test = y[:a], y[a:a+b], y[a+b:]
        
        # ## init loaders
        # self.train = DataLoader(TensorDataset(tc.tensor(x_train, dtype=tc.float32), tc.tensor(y_train)),
        #                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # self.val = DataLoader(TensorDataset(tc.tensor(x_val, dtype=tc.float32), tc.tensor(y_val)),
        #                       batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # self.test = DataLoader(TensorDataset(tc.tensor(x_test, dtype=tc.float32), tc.tensor(y_test)),
        #                        batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        ## init loaders
        ds = NormalDataset(sample_size['train'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ds = NormalDataset(sample_size['val'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        ds = NormalDataset(sample_size['test'], dist_params['mu'], dist_params['sig'], dim, dist_params['slope'])
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)        
        
        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        # ## plot data distributions
        # os.makedirs(os.path.dirname(fn), exist_ok=True)
        # with PdfPages(fn + '.pdf') as pdf:
        #     plt.figure(1)
        #     plt.clf()
        #     x = np.arange(-10, 10, 0.01)
        #     h1 = plt.plot(x, norm.pdf(x, dist_params['mu'], dist_params['sig']), 'r', label=r'$p(x)$')[0]
        #     h2 = plt.plot(x, 1.0/(1.0 + np.exp(-x*dist_params['slope'])), 'b', label=r'$p(y|x)$')[0]
        #     plt.grid('on')
        #     plt.xlabel('x', fontsize=fontsize)
        #     plt.ylabel('probability', fontsize=fontsize)
        #     plt.legend(handles=[h1, h2], fontsize=fontsize)

        #     plt.savefig(fn, bbox_inches='tight')
        #     pdf.savefig(bbox_inches='tight')



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
