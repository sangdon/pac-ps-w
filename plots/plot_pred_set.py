import os, sys
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    ## parameters
    fn_stats_list = [
        'snapshots/exp_normal/stats_pred_set.pk',
        'snapshots/exp_normal_shift/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_wscp/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_singlebin/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_worstbinopt/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_mgf/stats_pred_set.pk',
    ]
    name_list = [
        r'desired',
        r'PS-IID',
        r'WSCP',
        r'Thm1',
        r'Thm3',
        r'MGF',
    ]    
    fontsize = 15
    fig_root = 'snapshots/figs_normal_synthetic'

    ## init
    os.makedirs(fig_root, exist_ok=True)
    
    ## load
    stats_list = [pickle.load(open(fn, 'rb')) for fn in fn_stats_list]

    ## plot error
    error_list = [stat['error_test'].cpu().mean().numpy() for stat in stats_list]
    
    assert(all([stats_list[0]['eps'] == e for e in [stat['eps'] for stat in stats_list]]))
    assert(all([stats_list[0]['delta'] == e for e in [stat['delta'] for stat in stats_list]]))
    assert(all([stats_list[0]['n'] == e for e in [stat['n'] for stat in stats_list]]))
    eps = stats_list[0]['eps']
    delta = stats_list[0]['delta']
    n = stats_list[0]['n']
    fn = os.path.join(fig_root, 'plot_error_n_%d_eps_%f_delta_%f'%(n, eps, delta))
    
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.bar(np.arange(len(error_list)), error_list, width=0.3, color='r', edgecolor='k')
        h = plt.hlines(eps, -0.5, 0.5+len(error_list)-1, colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        plt.gca().tick_params(labelsize=fontsize*0.75)
        plt.gca().set_xticks(np.arange(len(error_list)))
        plt.gca().set_xticklabels(name_list, fontsize=int(fontsize*0.75))
        plt.ylabel('error', fontsize=fontsize)
        plt.grid('on')
        plt.legend(handles=[h], fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

    ## plot
    size_list = [stat['size_test'].cpu().numpy() for stat in stats_list]
    size_list = [s + np.random.normal(0, 1e-6, s.shape) for s in size_list] ## add very small noise to avoid visualization error of the box plot
    eps = stats_list[0]['eps']
    delta = stats_list[0]['delta']
    n = stats_list[0]['n']
    fn = os.path.join(fig_root, 'plot_size_n_%d_eps_%f_delta_%f'%(n, eps, delta))

    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.boxplot(size_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        plt.gca().set_xticklabels(name_list)
        plt.xticks(fontsize=int(fontsize*0.75))
        plt.yticks(fontsize=int(fontsize*0.75))
        plt.grid('on')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
