import os, sys
from learning import *
import numpy as np
import pickle
import types
import itertools
import scipy

import torch as tc
from .util import *

def geb_VC(delta, n, d=1.0):
    n = float(n)
    g = np.sqrt(((np.log((2*n)/d) + 1.0) * d + np.log(4/delta))/n)
    return g

def geb_iw_finite(delta, m, n_C, M, d2_max):
    m, n_C, M, d2_max = float(m), float(n_C), float(M), float(d2_max)
    g = 2.0*M*(np.log(n_C) + np.log(1.0/delta)) / 3.0 / m + np.sqrt( 2.0*d2_max*(np.log(n_C) + np.log(1.0/delta))/m )
    return g


def log_factorial(n):

    #log_f = tc.arange(n, 0, -1).float().log().sum()
    log_f = np.sum(np.log(np.arange(n, 0, -1.0)))
    
    return log_f


def log_n_choose_k(n, k):
    if k == 0:
        #return tc.tensor(1)
        return 1
    else:
        #res = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
        #res = tc.arange(n, n-k, -1).float().log().sum() - log_factorial(k)
        res = np.sum(np.log(np.arange(n, n-k, -1.0))) - log_factorial(k)
        return res

    
def half_line_bound_upto_k(n, k, eps):
    assert(eps > 0.0)
    ubs = []
    #eps = tc.tensor(eps)
    for i in np.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        #log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        #ubs.append(log_ub.exp().unsqueeze(0))
        log_ub = bc_log + np.log(eps)*i + np.log(1.0-eps)*(n-i)
        ubs.append([np.exp(log_ub)])
    ubs = np.concatenate(ubs)
    ub = np.sum(ubs)
    return ub


def binedges_equalmass(x, n_bins):
    n = len(x)
    return np.interp(np.linspace(0, n, n_bins + 1),
                     np.arange(n),
                     np.sort(x))



class PredSetConstructor(BaseLearner):
    def __init__(self, model, params=None, model_iw=None, iw_max=None, name_postfix=None):
        super().__init__(model, params, name_postfix)
        self.mdl_iw = model_iw
        self.iw_max = iw_max
        
        if params:
            base = os.path.join(
                params.snapshot_root,
                params.exp_name,
                f"model_params{'_'+name_postfix if name_postfix else ''}_n_{self.mdl.n}_eps_{self.mdl.eps}_delta_{self.mdl.delta}")
            self.mdl_fn_best = base + '_best'
            self.mdl_fn_final = base + '_final'
            self.mdl.to(self.params.device)
        else:
            self.params = types.SimpleNamespace(device=tc.device('cpu'))

        
    def _compute_error_permissive_VC(self, eps, delta, n):
        g = geb_VC(delta, n)    
        error_per = eps - g
        return round(error_per*n) if error_per >= 0.0 else None

    
    # def _compute_error_permissive_iw_finite(self, eps, delta, n, n_C, M, d2_max):
    #     g = geb_iw_finite(delta, n, n_C, M, d2_max)
    #     error_per = eps - g
    #     print(eps, g)
    #     # return ratio
    #     return error_per if error_per >= 0.0 else None

    
    def _compute_error_permissive_direct(self, eps, delta, n):
        k_min = 0
        k_max = n
        bnd_min = half_line_bound_upto_k(n, k_min, eps)
        if bnd_min > delta:
            return None
        assert(bnd_min <= delta)
        k = n
        while True:
            # choose new k
            k_prev = k
            #k = (T(k_min + k_max).float()/2.0).round().long().item()
            k = round(float(k_min + k_max)/2.0)
        
            # terinate condition
            if k == k_prev:
                break
        
            # check whether the current k satisfies the condition
            bnd = half_line_bound_upto_k(n, k, eps)
            if bnd <= delta:
                k_min = k
            else:
                k_max = k

        # confirm that the solution satisfies the condition
        k_best = k_min
        assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
        #error_allow = float(k_best) / float(n)
        return k_best

    
    def _find_opt_T(self, ld, n, error_perm):
        nlogp = []
        for x, y in ld:
            x = to_device(x, self.params.device)
            y = to_device(y, self.params.device)
            #x, y = x.to(self.params.device), y.to(self.params.device)
            nlogp_i = self.mdl(x, y)
            nlogp.append(nlogp_i)
        nlogp = tc.cat(nlogp)
        assert(n == len(nlogp))
        nlogp_sorted = nlogp.sort(descending=True)[0]
        T_opt = nlogp_sorted[error_perm]

        return T_opt


    def _find_opt_T_enumerate(self, ld, n, error_perm, T_list=tc.linspace(0.0, 0.0001, 100)):
        print("!! assume classification since T_list =", T_list)
        raise NotImplementedError
    
        T_opt = None
        T_ori = self.mdl.T.data
        for T in T_list:
            self.mdl.T.data = -(T.log())
            self.mdl.to(self.params.device)
            ## compute error
            error, w = [], []
            for x, y in ld:
                x = to_device(x, self.params.device)
                y = to_device(y, self.params.device)
                #x, y = x.to(self.params.device), y.to(self.params.device)
                member = self.mdl.membership(x, y)
                error.append((member == False).float())
                if self.mdl_iw is not None:
                    w.append(self.mdl_iw(x))
                    
            if self.mdl_iw is None:
                error = tc.cat(error)
                error = error.mean()
            else:
                error, w = tc.cat(error), tc.cat(w)
                assert(error.shape == w.shape)
                error = (error*w).mean()
            print('T = %f, error = %f, error_perm = %f'%(T, error, error_perm))

            if error <= error_perm:
                T_opt = -(T.log())
            else:
                break
        self.mdl.T.data = T_ori
        return T_opt

                
    def train(self, ld):
        n, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        
        ## compute permissive error
        if self.params.bnd_type == 'VC':
            error_permissive = self._compute_error_permissive_VC(eps, delta, n)
        elif self.params.bnd_type == 'direct':
            error_permissive = self._compute_error_permissive_direct(eps, delta, n)
        # elif self.params.bnd_type == 'iw_finite':
        #     raise NotImplementedError
        #     print("!! TODO: generalize code, n_C = 100")
        #     error_permissive = self._compute_error_permissive_iw_finite(eps, delta, n, n_C=100, M=5, d2_max=5) ##TODO: generalize
        #     print('iw_finite: error_permissive', error_permissive)
        else:
            raise NotImplementedError
        
        if error_permissive is None:
            print("## construction failed: too strict parameters")
            return False

        ## find the optimal T
        # ##TODO: generalize
        # if self.params.bnd_type == 'iw_finite':
        #     T_opt = self._find_opt_T_enumerate(ld, n, error_permissive)
        # else:
        #     T_opt = self._find_opt_T(ld, n, error_permissive)
        
        
        T_opt = self._find_opt_T(ld, n, error_permissive)
        self.mdl.T.data = T_opt
        self.mdl.to(self.params.device)
        print(f"error_permissive = {error_permissive}, T_opt = {T_opt}")

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    def test(self, ld, ld_name, verbose=False):

        ## compute set size and error
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'stats_pred_set.pk')
        if os.path.exists(fn) and not self.params.rerun:
            res = pickle.load(open(fn, 'rb'))
            error = res['error_test']
            size = res['size_test']
        else:
            size, error = [], []
            for x, y in ld:
                size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                size.append(size_i)
                error.append(error_i)
            size, error = tc.cat(size), tc.cat(error)
            
            pickle.dump({'error_test': error, 'size_test': size, 'n': self.mdl.n, 'eps': self.mdl.eps, 'delta': self.mdl.delta}, open(fn, 'wb'))

        if verbose:
            mn = size.min()
            Q1 = size.kthvalue(int(round(size.size(0)*0.25)))[0]
            Q2 = size.median()
            Q3 = size.kthvalue(int(round(size.size(0)*0.75)))[0]
            mx = size.max()
            av = size.mean()
            print(
                f'[test: {ld_name}, n = {self.mdl.n}, eps = {self.mdl.eps:.2e}, delta = {self.mdl.delta:.2e}, T = {(-self.mdl.T.data).exp():.5f}] '
                f'error = {error.mean():.4f}, min = {mn}, 1st-Q = {Q1}, median = {Q2}, 3rd-Q = {Q3}, max = {mx}, mean = {av:.2f}'
            )

            ## plot results

            
        return size.mean(), error.mean()

