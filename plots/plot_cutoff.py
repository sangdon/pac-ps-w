import os, sys
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from torch import nn
from main_cls_syn import *
import model
import learning

if __name__ == '__main__':
    ## parameters
    fn_stats_list = [
        'snapshots/exp_normal/stats_pred_set.pk',
        'snapshots/exp_normal_shift/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_singlebin/stats_pred_set.pk',
    ]
    name_list = [
        r'desired',
        r'shift + PSIID',
        r'shift + PS1BIN',
    ]
    fontsize = 15
    fig_root = 'snapshots/figs_normal_synthetic'

    ## init
    os.makedirs(fig_root, exist_ok=True)
    
    ## loader
    ds_tar = data.NormalShift(
        batch_size=100,
        sample_size={'train': 10000, 'val': 10000, 'test': 10000},
        dist_params={'mu': 0.0, 'sig': 1.0, 'slope': 5.0},
    )

    ## models
    mdl = SynModel()
    mdl_predset = model.PredSetCls(mdl, 0.01, 1e-5, 10000)

    ## get stats
    logTs = tc.arange(0.3, 1.0, 0.1).log()
    size_list, error_list, T_list = [], [], []
    for logT in logTs:
        mdl_predset.T.data = -logT        
        l = learning.PredSetCovConstructor_worstbinopt(mdl_predset, None, model_iw=None)
        size_i, error_i = l.test(ds_tar.test, 'tar test')
        size_list.append(size_i.detach().cpu().numpy())
        error_list.append(error_i.detach().cpu().numpy())
        T_list.append(logT.exp().detach().cpu().numpy())

    ## plot
    fn = os.path.join(fig_root, 'plot_tradeoff')
    
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()

        h1 = plt.plot(T_list, size_list, 'k--', label='size')[0]
        h2 = plt.plot(T_list, error_list, 'k-', label='error')[0]
        
        plt.gca().tick_params(labelsize=fontsize*0.75)
        plt.xlabel('T', fontsize=fontsize)
        plt.grid('on')
        plt.legend(handles=[h1, h2], fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

