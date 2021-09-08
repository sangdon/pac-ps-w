import os, sys
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    fn_stats_baseline = 'snapshots/imagenet_intensity_shift_baseline/stats_pred_set.pk'
    fn_stats_ours = 'snapshots/imagenet_intensity_shift/stats_pred_set.pk'
    fontsize = 15
    
    ## load
    stats_baseline = pickle.load(open(fn_stats_baseline, 'rb'))
    stats_ours = pickle.load(open(fn_stats_ours, 'rb'))

    ## plot error
    fn = 'plot_error'
    error_baseline = stats_baseline['error_test'].cpu().mean().numpy()
    error_ours = stats_ours['error_test'].cpu().mean().numpy()
    assert(stats_baseline['eps'] == stats_ours['eps'])
    eps = stats_baseline['eps']
    labels = ['baseline', 'ours']

    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.bar([0.0, 1.0], [error_baseline, error_ours], width=0.3, color='r', edgecolor='k')
        h = plt.hlines(eps, -0.5, 1.5, colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        plt.gca().tick_params(labelsize=fontsize*0.75)
        plt.gca().set_xticks([0.0, 1.0])
        plt.gca().set_xticklabels(labels, fontsize=fontsize)
        plt.ylabel('error', fontsize=fontsize)
        plt.grid('on')
        plt.legend(handles=[h], fontsize=fontsize)
        plt.savefig(fn, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
    

    ## plot
    fn = 'plot_size'
    size_baseline = stats_baseline['size_test'].cpu().numpy()
    size_ours = stats_ours['size_test'].cpu().numpy()
    labels = ['baseline', 'ours']

    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.boxplot([size_baseline, size_ours], whis=np.inf,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        plt.gca().set_xticklabels(labels)
        plt.xticks(fontsize=int(fontsize*0.75))
        plt.yticks(fontsize=int(fontsize*0.75))
        plt.grid('on')
        plt.savefig(fn, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
