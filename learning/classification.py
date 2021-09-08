import os, sys
import time

from learning import *
#from uncertainty import compute_ece

class ClsLearner(BaseLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_01
        self.loss_fn_test = loss_01
        

    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        error, *_ = super().test(ld, mdl, loss_fn)
        ece = compute_ece(self.mdl if mdl is None else mdl, ld, self.params.device)
        error_label = compute_error_per_label(
            ld,
            self.mdl if mdl is None else mdl,
            loss_fn if loss_fn else self.loss_fn_test,
            self.params.device)
        
        if verbose:
            print('[test%s, %f secs.] classificaiton error = %.2f%%, calibration error = %.2f%%'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0, ece*100.0))
        # for i, e in enumerate(error_label):
        #     print(f'[class = {i}] error = {e}')


        return error, ece, error_label


