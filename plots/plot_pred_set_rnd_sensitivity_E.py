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
    #parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--dataset', type=str, default='domainnet_src_DomainNetAll_tar_DomainNetSketch_da_iwcal')
    parser.add_argument('--trueiw', action='store_true')
    parser.add_argument('--m', type=int, default=50000)
    parser.add_argument('--eps', type=str, default='0.1')
    parser.add_argument('--delta', type=str, default='1e-5')
    parser.add_argument('--fontsize', type=int, default=20)
    parser.add_argument('--figsize', type=float, nargs=2, default=[6.4*1.5, 4.8])
    parser.add_argument('--E', type=float, nargs='*', default=['0', '0.001', '0.005', '0.01'])
    
    args = parser.parse_args()
    
    fontsize = args.fontsize
    fig_root = f'{args.snapshot_root}/figs/{args.dataset.split("_")[0]}/{args.dataset}_sensitivity_E'

    exp_name_list = ['exp_' + args.dataset + '_' + 'worst_rejection' + f'_m_{args.m}_eps_{args.eps}_delta_{args.delta}_E_{E}' for E in args.E]
    name_list = [E for E in args.E]

    ## init
    os.makedirs(fig_root, exist_ok=True)
    
    ## load
    stats_rnd_fn_list = [glob.glob(os.path.join(args.snapshot_root, exp_name+'_expid_*', 'stats_pred_set.pk')) for exp_name in exp_name_list]

    
    # for l in stats_rnd_fn_list:
    #     for fn in l:
    #         print(fn)
    #         x = pickle.load(open(fn, 'rb'))
    # sys.exit()
    stats_list = [[pickle.load(open(fn, 'rb')) for fn in l] for l in stats_rnd_fn_list]

    for E, fn_stats in zip(args.E, stats_rnd_fn_list):
        print(f'[E = {E}] #exps = {len(fn_stats)}')
    
    ## sanity check
    for i, l in enumerate(stats_list):
        if len(l) > 0:
            i_val = i
    eps = stats_list[i_val][0]['eps'].item()
    delta = stats_list[i_val][0]['delta'].item()
    n = stats_list[i_val][0]['n'].item()

    assert(all([eps == e for e in [s['eps'] for stat in stats_list for s in stat]]))
    #assert(all([abs(delta - d) < 1e-8 or abs(delta/2 - d) < 1e-8 for d in [s['delta'] for stat in stats_list for s in stat]]))
    assert(all([n == n for n in [s['n'] for stat in stats_list for s in stat]]))
    assert(abs(eps - float(args.eps)) < 1e-8)
    #assert(abs(delta - float(args.delta)) < 1e-8)
    assert(n == args.m)
    
    print('eps = %f, delta = %f, n = %d'%(eps, delta, n))

    
    ## plot error
    error_list = [np.array([s['error_test'].cpu().mean().numpy() + np.random.rand()*1e-16 for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_error_rnd_n_%d_eps_%f_delta_%f'%(n, eps, delta))
    print(fn)
    with PdfPages(fn + '.pdf') as pdf:
        plt.figure(1, figsize=args.figsize)
        plt.clf()
        plt.boxplot(error_list, whis=np.inf, showmeans=True,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        h = plt.hlines(eps, 0.5, 0.5+len(error_list), colors='k', linestyles='dashed', label='$\epsilon = %.2f$'%(eps))
        #plt.gca().tick_params(labelsize=fontsize)
        #plt.gca().set_xticks(np.arange(len(error_list)))
        plt.gca().set_xticklabels(name_list, fontsize=fontsize)
        plt.xlabel('E', fontsize=fontsize)
        plt.ylabel('prediction set error', fontsize=fontsize)
        plt.ylim(bottom=0.0)
        plt.gca().tick_params('x', labelsize=fontsize*0.75)
        plt.gca().tick_params('y', labelsize=fontsize*0.75)
        plt.grid('on')
        plt.legend(handles=[h], fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

    ## plot
    print('plot only the first experiment')
    size_list = [np.array([s['size_test'][0].cpu() for s in stat]) for stat in stats_list]
    fn = os.path.join(fig_root, 'plot_size_rnd_n_%d_eps_%f_delta_%f'%(n, eps, delta))
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
