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
        'snapshots/exp_normal_shift_iw_worstbinopt_n_10000/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_worstbinopt_n_20000/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_worstbinopt_n_30000/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_worstbinopt_n_40000/stats_pred_set.pk',
        'snapshots/exp_normal_shift_iw_worstbinopt_n_50000/stats_pred_set.pk',
    ]
    name_list = [
        r'$10,000$',
        r'$20,000$',
        r'$30,000$',
        r'$40,000$',
        r'$50,000$',
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
    
    eps = stats_list[0]['eps']
    delta = stats_list[0]['delta']
    fn = os.path.join(fig_root, 'plot_error_various_n_eps_%f_delta_%f'%(eps, delta))
    
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.bar(np.arange(len(error_list)), error_list, width=0.3, color='r', edgecolor='k')
        h = plt.hlines(eps, -0.5, 0.5+len(error_list)-1, colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        plt.gca().tick_params(labelsize=fontsize*0.75)
        plt.gca().set_xticks(np.arange(len(error_list)))
        plt.gca().set_xticklabels(name_list, fontsize=int(fontsize*0.75))
        plt.xlabel(r'sample size $m$', fontsize=fontsize)
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
    fn = os.path.join(fig_root, 'plot_size_various_n_eps_%f_delta_%f'%(eps, delta))

    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.boxplot(size_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        plt.gca().set_xticklabels(name_list)
        plt.xticks(fontsize=int(fontsize*0.75))
        plt.yticks(fontsize=int(fontsize*0.75))
        plt.xlabel(r'sample size $m$', fontsize=fontsize)
        plt.ylabel('set size', fontsize=fontsize)
        plt.grid('on')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
