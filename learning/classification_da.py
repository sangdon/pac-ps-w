import os, sys
import time
import math

from learning import *


class ClsDALearner(ClsLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        ## schedule regularization paraeter
        self.reg_param_adv = 0.0 
        self.loss_fn_train = lambda *args, **kwargs: loss_xe_adv(*args, **kwargs, reg_param_adv=self.reg_param_adv)

        
    def _train_epoch_begin(self, i_epoch):
        super()._train_epoch_begin(i_epoch)

        # schedule adversarial regularization parameter        
        i_rate = (i_epoch-1) / self.params.n_epochs
        reg_param_adv_final = 1.0
        self.reg_param_adv = reg_param_adv_final * (2.0 / (1 + math.exp(-10.0 * i_rate)) - 1.0)



