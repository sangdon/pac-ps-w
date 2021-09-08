import os, sys
import numpy as np
import pickle
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    ## parameters
    snapshot_root = 'snapshots'
    exp_name_list = [
        'exp_normal',
        'exp_normal_shift',
        'exp_normal_shift_iw_wscp',
        'exp_normal_shift_iw_singlebin',
        'exp_normal_shift_iw_worstbinopt',
        'exp_normal_shift_iw_worstbinopt_eqmassbins',
        'exp_normal_shift_iw_mgf',
        'exp_normal_shift_iw_mgf_eqmassbins',
    ]

    name_list = [
        r'desired',
        r'PS-IID',
        r'WSCP',
        r'Thm1',
        r'Thm3',
        r'Thm3-E',
        r'MGF',
        r'MGF-E',
    ]    
    fontsize = 15
    fig_root = 'snapshots/figs_normal_synthetic'

    ## init
    os.makedirs(fig_root, exist_ok=True)
    
    ## load
    stats_rnd_fn_list = [glob.glob(os.path.join(snapshot_root, exp_name+'_expid_*', 'stats_pred_set.pk')) for exp_name in exp_name_list]
    stats_list = [[pickle.load(open(fn, 'rb')) for fn in l] for l in stats_rnd_fn_list]

    ## sanity check
    eps = stats_list[0][0]['eps']
    delta = stats_list[0][0]['delta']
    n = stats_list[0][0]['n']

    assert(all([eps == e for e in [s['eps'] for stat in stats_list for s in stat]]))
    assert(all([delta == d for d in [s['delta'] for stat in stats_list for s in stat]]))
    assert(all([n == n for n in [s['n'] for stat in stats_list for s in stat]]))
    print('eps = %f, delta = %f, n = %d'%(eps, delta, n))
    
    ## plot error
    error_list = [np.array([s['error_test'].cpu().mean().numpy() for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_error_rnd_n_%d_eps_%f_delta_%f'%(n, eps, delta))
    
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.boxplot(error_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        h = plt.hlines(eps, 0.5, 0.5+len(error_list), colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        plt.gca().tick_params(labelsize=fontsize*0.75)
        #plt.gca().set_xticks(np.arange(len(error_list)))
        plt.gca().set_xticklabels(name_list, fontsize=int(fontsize*0.75))
        plt.ylabel('error', fontsize=fontsize)
        plt.grid('on')
        plt.legend(handles=[h], fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

    ## plot
    size_list = [np.array([s['size_test'].cpu().mean() for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_size_rnd_n_%d_eps_%f_delta_%f'%(n, eps, delta))

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
