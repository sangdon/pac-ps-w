import os, sys
import time

from learning import *

class ClsLearner(BaseLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_01
        self.loss_fn_test = loss_01
        

    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        error, *_ = super().test(ld, mdl, loss_fn)
        
        if verbose:
            print('[test%s, %f secs.] classificaiton error = %.2f%%'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0))
        
        return error,


