import os, sys
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':

    ## parameters
    fontsize = 15
    width = 0.15  # the width of the bars
    
    name_list = [
        r'$\mathcal{M}\rightarrow\mathcal{M}$',
        r'$\mathcal{U}\rightarrow\mathcal{M}$',
        r'$\mathcal{M}\rightarrow\mathcal{U}$',
        r'$\mathcal{S}\rightarrow\mathcal{M}$',
        r'$\mathcal{M}\rightarrow\mathcal{S}$',
        r'$\mathcal{A}\rightarrow\mathcal{W}$',
        r'$\mathcal{D}\rightarrow\mathcal{A}$',
        r'$\mathcal{W}\rightarrow\mathcal{A}$',
    ]
    fn_plot = 'plot_ece_cal_cov'
    
    ## ece in percent
    ece_softmax = [0.83, 42.43, 21.26, 33.94, 59.96, 43.82, 35.09, 27.48]
    ece_softmax_std = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ece_temp = [0.18, 36.89, 12.16, 28.63, 10.28, 26.69, 66.21, 54.59]
    ece_temp_std = [0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.0, 0.0]
    
    ece_iw_temp = [0.27, 7.68, 15.92, 26.08, 25.47, 21.82, 65.89, 54.20]
    ece_iw_temp_std = [0.0, 1.80, 1.38, 0.46, 8.98, 1.20, 0.01, 0.01]

    ece_fl_temp = [0.24, 17.08, 11.47, 23.37, 3.90, 17.29, 21.39, 25.21]
    ece_fl_temp_std = [0.04, 1.01, 1.36, 0.42, 1.96, 2.28,  3.17, 1.38]

    ece_fl_iw_temp = [0.21, 10.58, 10.85, 21.59, 8.57, 13.49, 21.25, 19.85]
    ece_fl_iw_temp_std = [0.02, 4.67, 1.45, 5.40, 5.60, 1.62, 2.84,  6.54]
    

    ## plot ece
    with PdfPages(fn_plot + '.pdf') as pdf:
        plt.figure(1, figsize=[6.4*2, 4.8])
        plt.clf()

        x = np.arange(len(name_list))
        
        ## ece bars
        hs = []
        hs.append(plt.bar(x - width*2, ece_softmax, width, yerr=ece_softmax_std, error_kw={'ls': '--', 'capsize': 3}, label='softmax'))
        hs.append(plt.bar(x - width*1, ece_temp, width, yerr=ece_temp_std, error_kw={'ls': '--', 'capsize': 3}, label='temp.', color='C3'))
        hs.append(plt.bar(x, ece_iw_temp, width, yerr=ece_iw_temp_std, error_kw={'ls': '--', 'capsize': 3}, label='IW+temp. (ours)', color='C2', alpha=0.5))
        hs.append(plt.bar(x + width*1, ece_fl_temp, width, yerr=ece_fl_temp_std, error_kw={'ls': '--', 'capsize': 3}, label='FL+temp. (ours)', color='C2', alpha=0.7))
        hs.append(plt.bar(x + width*2, ece_fl_iw_temp, width, yerr=ece_fl_iw_temp_std, error_kw={'ls': '--', 'capsize': 3}, label='FL+IW+temp. (ours)', color='C2'))
        hs[-1][-3].set_linestyle('--') 
        ## std bar

        ## beautify
        plt.ylabel('ECE (%)', fontsize=fontsize)
        plt.ylim((0.0, 80.0))
        plt.xticks(x)
        plt.gca().set_xticklabels(name_list, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.grid('on')
        plt.legend(handles=hs, fontsize=fontsize)
        
        plt.savefig(fn_plot, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')
    
