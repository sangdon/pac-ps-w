import os, sys
import numpy as np
import pickle
import glob
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning')
    ## parameters

    ## meta args
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--dataset', type=str, default='domainnet_src_DomainNetAll_tar_DomainNetSketch_da')
    parser.add_argument('--trueiw', action='store_true')
    parser.add_argument('--m', type=int, default=50000)
    parser.add_argument('--eps', type=str, default='0.1')
    parser.add_argument('--delta', type=str, default='1e-5')
    parser.add_argument('--fontsize', type=int, default=20)
    parser.add_argument('--figsize', type=float, nargs=2, default=[5.0, 4.8])
    args = parser.parse_args()
    
    fontsize = args.fontsize
    fig_root = f'{args.snapshot_root}/figs/{args.dataset.split("_")[0]}/{args.dataset}'

    if args.trueiw:
        method_name_list = [
            ('naive', 'PS'),
            ('wscp', 'WSCI'),
            ('maxiw', 'PS-C'),
            ('rejection', 'PS-R'),
        ]
    else:    
         method_name_list = [
             ('naive', 'PS'),
             ('wscp', 'WSCI'),
             ('maxiw', 'PS-C'),
             ('worst_rejection', 'PS-W'),
        ]
    exp_name_list = ['exp_' + args.dataset + '_' + n[0] + f'_m_{args.m}_eps_{args.eps}_delta_{args.delta}' for n in method_name_list]
    name_list = [n[1] for n in method_name_list]

    ## init
    os.makedirs(fig_root, exist_ok=True)
    
    ## load
    stats_rnd_fn_list = [glob.glob(os.path.join(args.snapshot_root, exp_name+'_expid_*', 'stats_pred_set.pk')) for exp_name in exp_name_list]
    stats_list = [[pickle.load(open(fn, 'rb')) for fn in l] for l in stats_rnd_fn_list]

    for n, fn_stats in zip(method_name_list, stats_rnd_fn_list):
        print(f'[method = {n[1]}] #exps = {len(fn_stats)}')
    
    ## sanity check
    i_val = None
    for i, l in enumerate(stats_list):
        if len(l) > 0:
            i_val = i
    assert i_val is not None, f'No data to plot'
        
    eps = stats_list[i_val][0]['eps'].item()
    delta = stats_list[i_val][0]['delta'].item()
    n = stats_list[i_val][0]['n'].item()    

    assert(all([eps == e for e in [s['eps'] for stat in stats_list for s in stat]]))
    assert(all([n == n for n in [s['n'] for stat in stats_list for s in stat]]))
    assert(abs(eps - float(args.eps)) < 1e-8)
    assert(n == args.m)
    
    print('eps = %f, delta = %f, n = %d'%(eps, delta, n))

    
    ## plot error
    error_list = [np.array([s['error_test'].cpu().mean().numpy() for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_error_rnd_n_%d_eps_%f_delta_%f'%(args.m, float(args.eps), float(args.delta)))
    print(fn)
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1, figsize=args.figsize)
        plt.clf()
        plt.boxplot(error_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        h = plt.hlines(eps, 0.5, 0.5+len(error_list), colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        plt.gca().set_xticklabels(name_list, fontsize=fontsize)
        plt.ylabel('prediction set error', fontsize=fontsize)
        plt.ylim(bottom=0.0)
        plt.gca().tick_params('y', labelsize=fontsize*0.75)
        plt.grid('on')
        plt.legend(handles=[h], fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

    ## plot
    print('plot only the first experiment')
    size_list = [np.array([s['size_test'][0].cpu() for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_size_rnd_n_%d_eps_%f_delta_%f'%(args.m, float(args.eps), float(args.delta)))
    print(fn)
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1, figsize=args.figsize)
        plt.clf()
        plt.boxplot(size_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        plt.gca().set_xticklabels(name_list)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('size', fontsize=fontsize)
        plt.ylim(bottom=0.0)
        plt.grid('on')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
    print()
