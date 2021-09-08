import os, sys
import time

from learning import *
from uncertainty import *


class TempScalingRegLearner(RegLearner):
    def __init__(self, mdl, params=None, name_postfix='cal_temp_reg'):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = lambda x, y, model, reduction, device: loss_nll(x, y, model, reduction, device, self.params.normalizer)
        self.loss_fn_val = loss_nll
        self.loss_fn_test = loss_nll
        self.T_min = 1e-9
        
        
    # def _train_epoch_batch_end(self, i_epoch):
    #     #[T.data.clamp_(self.T_min) for T in self.mdl.parameters()]
    #     print("T =", [T.exp() for T in self.mdl.parameters()])
        

    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        
        ## compute regression error
        error, *_ = super().test(ld, mdl, loss_fn)

        if verbose:
            print('[test%s, %f secs.] test nll = %.2f'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error))

        return error,

