import os, sys
import numpy as np
import math

import torch as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def compute_conf(mdl, ld, device):
    ph_list = []
    mdl = mdl.to(device)
    for x, y in ld:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            out = mdl(x)
        ph = out['ph_cal']
        ph_list.append(ph.cpu())
    ph_list = tc.cat(ph_list)
    return ph_list


def plot_induced_dist_iw(f_nll, w_lower, w_upper, fn, n_bins=15, fontsize=15):

    os.makedirs(os.path.dirname(fn), exist_ok=True)

    print(w_upper)
    
    p_yx = np.exp(-f_nll)
    p_yx_hist, p_yx_bin_edges = np.histogram(p_yx, bins=n_bins, density=False)
    p_yx_hist_norm = p_yx_hist / np.sum(p_yx_hist)

    w_lower_bin, w_upper_bin = [], []
    for i, (l, u) in enumerate(zip(p_yx_bin_edges[:-1], p_yx_bin_edges[1:])):
        if i == len(p_yx_bin_edges)-2:
            idx = (p_yx >= l) & (p_yx <= u)
        else:            
            idx = (p_yx >= l) & (p_yx < u)
        w_lower_bin.append(w_lower[idx].mean())
        w_upper_bin.append(w_upper[idx].mean())

        
    with PdfPages(fn + '.pdf') as pdf: 
        h_list = []
        plt.figure(1)
        plt.clf()
        fig, ax1 = plt.subplots()
        # plot a induced distribution
        h = ax1.bar((p_yx_bin_edges[1:] + p_yx_bin_edges[:-1])/2.0, p_yx_hist_norm,
                    width=(p_yx_bin_edges[1] - p_yx_bin_edges[0])*0.7, label='source distr.', alpha=0.7)
        h_list.append(h)
        
        # plot iw
        ax2 = ax1.twinx()
        ax2.plot((p_yx_bin_edges[1:] + p_yx_bin_edges[:-1])/2.0, [(l+u)/2.0 for l, u in zip(w_lower_bin, w_upper_bin)], color='k')[0]        
        h = ax2.fill_between((p_yx_bin_edges[1:] + p_yx_bin_edges[:-1])/2.0, w_lower_bin, w_upper_bin, color='r', alpha=0.3, label='IW')
        h_list.append(h)
        
        # beautify
        ax1.grid('on')
        ax1.set_xlabel('$\hat{f}(x, y)$', fontsize=fontsize)
        ax1.set_ylabel('distribution', fontsize=fontsize)
        ax2.set_ylabel('importance weight (IW)', fontsize=fontsize)
        ax1.set_ylim((0.0, 1.0))
        plt.legend(handles=h_list, fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')


def plot_histbin(bins, ch, lower, upper, n, fn):

    with PdfPages(fn + '.pdf') as pdf: 

        plt.figure(1)
        plt.clf()

        # number of points
        ph = bins[0:-1] + (bins[1:] - bins[0:-1]) / 2.0
        w = (bins[1] - bins[0]) * 0.75
        n_normalized = n / np.sum(n)
        plt.bar(ph, n_normalized, color='r', edgecolor='k', alpha=0.3, width=w)    
        # error bar
        plt.errorbar(ph, ch, yerr=[ch-lower, upper-ch], fmt='ks')
        # beautify
        plt.grid('on')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))
        plt.xlabel('ph')
        plt.ylabel('ch')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')


def plot_induced_dist(p_yx, q_yx, iw, fn, fontsize=15, n_bins=20):

    r_max = max(p_yx.max(), q_yx.max())
    
    hist_p, bin_edges_p = np.histogram(p_yx, bins=n_bins, range=(0.0, r_max), density=False)
    hist_p_norm = hist_p/hist_p.sum()

    hist_q, bin_edges_q = np.histogram(q_yx, bins=n_bins, range=(0.0, r_max), density=False)
    hist_q_norm = hist_q/hist_q.sum()

    #iw = hist_q_norm / hist_p_norm
    iw_plot, iw_mean_plot = [], []
    for i, (l, u) in enumerate(zip(bin_edges_p[:-1], bin_edges_p[1:])):
        if i == len(bin_edges_p)-2:
            idx = (p_yx >= l) & (p_yx <= u)
            idx_mean = p_yx <= u
        else:            
            idx = (p_yx >= l) & (p_yx < u)
            idx_mean = p_yx < u
        iw_plot.append(iw[idx].mean())
        iw_mean_plot.append(iw[idx_mean].mean())
        
    ## plot
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with PdfPages(fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        fig, ax1 = plt.subplots()
        # plot source induced distribution
        h1 = ax1.bar((bin_edges_p[1:] + bin_edges_p[:-1])/2.0, hist_p_norm, width=(bin_edges_p[1] - bin_edges_p[0])*0.7, label='source', alpha=0.7)
        # plot target induced distribution
        h2 = ax1.bar((bin_edges_q[1:] + bin_edges_q[:-1])/2.0, hist_q_norm, width=(bin_edges_q[1] - bin_edges_q[0])*0.7, label='target', alpha=0.7)        
        # plot iw
        ax2 = ax1.twinx()
        h3 = ax2.plot((bin_edges_q[1:] + bin_edges_q[:-1])/2.0, iw_plot, 'k--', label='IW')[0]
        # plot cond. mean of IW
        h4 = ax2.plot((bin_edges_q[1:] + bin_edges_q[:-1])/2.0, iw_mean_plot, 'r-', label='cond. mean of IW')[0]
        
        # beautify
        ax1.grid('on')
        ax1.set_xlim((0.0, r_max))
        ax1.set_xlabel('p(y | x)', fontsize=fontsize)
        ax1.set_ylabel('probability', fontsize=fontsize)
        ax2.set_ylabel('importance weight (IW)', fontsize=fontsize)
        ax2.set_ylim((0.0, 10.0))
        plt.legend(handles=[h1, h2, h3, h4], fontsize=fontsize, loc='upper left')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

        
def plot_induced_dist_wrapper(ld_src, ld_tar, mdl, mdl_iw, device, fn):
    mdl, mdl_iw = mdl.to(device), mdl_iw.to(device)
    mdl, mdl_iw = mdl.eval(), mdl_iw.eval()

    def get_ph_yx(ld, mdl, mdl_iw, device):    
        ph_yx_list = []
        w_list = []
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            ph = mdl(x)['ph']
            if mdl_iw is not None:
                w = mdl_iw(x)
                w_list.append(w)
            ph_yx = ph.gather(1, y.view(-1, 1)).squeeze(1)
            ph_yx_list.append(ph_yx)
        ph_yx_list = tc.cat(ph_yx_list)
        ph_yx_list = ph_yx_list.cpu().detach().numpy()
        if len(w_list) > 0:
            w_list = tc.cat(w_list)
            w_list = w_list.cpu().detach().numpy()
        return ph_yx_list, w_list

    p_yx, iw = get_ph_yx(ld_src, mdl, mdl_iw, device)
    q_yx, _ = get_ph_yx(ld_tar, mdl, None, device)
    plot_induced_dist(p_yx, q_yx, iw, fn)


def plot_iw(iw, fn, fontsize=15, n_bins=20):

    ## plot
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with PdfPages(fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()

        plt.hist(iw, edgecolor='k')
        
        # beautify
        plt.grid('on')
        plt.xlabel('importance weight w(x)', fontsize=fontsize)
        plt.ylabel('count', fontsize=fontsize)
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')


def plot_iw_wrapper(ld_src, mdl_iw, device, fn):
    mdl_iw = mdl_iw.to(device).eval()

    ## get iws
    w_list = []
    for x, y in ld_src:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            w = mdl_iw(x)
        w_list.append(w)
    w_list = tc.cat(w_list)
    w_list = w_list.cpu().detach().numpy()

    plot_iw(w_list, fn)


def plot_wh_w(iw, dom_label, fn, fontsize=15, n_bins=20):

    _, bin_edges = np.histogram(iw, bins=n_bins)
    iw_max = bin_edges[-1]
    
    iw_est, iw_true, rate_src = [], [], []
    for i, (l, u) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == len(bin_edges)-2:
            idx = (iw>=l) & (iw<=u)
        else:
            idx = (iw>=l) & (iw<u)
        label_i = dom_label[idx]
        n_src = float(np.sum(label_i == 1))
        n_tar = float(np.sum(label_i == 0))
        iw_true.append(n_tar / n_src if n_src > 0 else np.inf)
        iw_est.append((l+u)/2.0)
        rate_src.append(n_src)
    rate_src = rate_src/np.sum(rate_src)
        
    ## plot
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with PdfPages(fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        
        fig, ax1 = plt.subplots()
        #plt.bar(iw_est, iw_true, width=(bin_edges[1]-bin_edges[0])*0.75, color='r', edgecolor='k')
        h1 = ax1.plot(iw_est, iw_true, 'rs--', label='estimated-true')[0]
        h2 = ax1.plot(np.arange(0, bin_edges[-1], 0.1), np.arange(0, bin_edges[-1], 0.1), 'k-', label='ideal')[0]

        ax2 = ax1.twinx()
        h3 = ax2.bar(iw_est, rate_src, width=(bin_edges[1]-bin_edges[0])*0.75, color='b', edgecolor='k', alpha=0.5, label='source rate')

        # beautify
        ax1.grid('on')
        ax1.set_xlabel('estimated IW', fontsize=fontsize)
        ax1.set_ylabel('coarsened true IW', fontsize=fontsize)
        ax2.set_ylabel('source rate', fontsize=fontsize)
        ax2.set_ylim((0, 1.0))
        # plt.xlim((0, np.ceil(iw_max)))
        # plt.ylim((0, np.ceil(iw_max)))
        plt.legend(handles=[h1, h2, h3], fontsize=fontsize, loc='upper center')
        plt.savefig(fn+'.png', bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')


def plot_wh_w_wrapper(ld_dom, mdl_iw, device, fn):
    mdl_iw = mdl_iw.to(device).eval()

    ## get iws
    w_list, dom_list = [], []
    for x, y in ld_dom:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            w = mdl_iw(x)
        w_list.append(w)
        dom_list.append(y)
    w_list, dom_list = tc.cat(w_list), tc.cat(dom_list)
    w_list, dom_list = w_list.cpu().detach().numpy(), dom_list.cpu().detach().numpy()

    plot_wh_w(w_list, dom_list, fn=fn)

    
    
def estimate_eff_sample_size(ld_src, mdl_iw, device):
    ## precompute w(x)
    w_list = []
    for x, y in ld_src:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            w_i = mdl_iw(x, y)
        w_list.append(w_i)
    w_list = tc.cat(w_list)
    m_eff = w_list.sum().pow(2.0).float() / w_list.pow(2.0).sum().float()
    m_eff = m_eff.floor().int()
    return m_eff.item()
    
    

def estimate_mean_worst_hoeffding(mean_emp, n, a, b, delta):
    err_est = math.sqrt(pow(b-a, 2) * math.log(1.0 / delta) / 2.0 / n)
    mean_worst = mean_emp + err_est
    return mean_worst


def estimate_mean_hoeffding(mean_emp, n, a, b, delta, ret_est_err=False):
    err_est = math.sqrt(pow(b-a, 2) * math.log(2.0 / delta) / 2.0 / n)
    if ret_est_err:
        return err_est
    else:
        interval = (mean_emp - err_est, mean_emp + err_est)
        return interval


#https://arxiv.org/pdf/0907.3740.pdf: thm4
def estimate_mean_worst_emp_bernstein(mean_emp, std_emp_unbiased, n, a, b, delta):
    t1 = std_emp_unbiased * math.sqrt(2.0 * math.log(2.0 / delta) / n)
    t2 = 7.0 * (b-a) * math.log(2.0 / delta) / 3.0 / (n - 1.0)
    err_est = t1 + t2
    mean_worst = mean_emp + err_est
    return mean_worst


def bci_clopper_pearson(k, n, alpha, two_side=True, use_R=False):
    if two_side:
        if use_R: # R is numerically better when alpha is small
            from rpy2.robjects.packages import importr
            stats = importr('stats')

            lo = stats.qbeta(alpha/2, int(k), int(n-k+1))[0]
            hi = stats.qbeta(1 - alpha/2, int(k+1), int(n-k))[0]
        else:
            from scipy import stats

            lo = stats.beta.ppf(alpha/2, k, n-k+1)
            hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
        
            lo = 0.0 if math.isnan(lo) else lo
            hi = 1.0 if math.isnan(hi) else hi
    
        return lo, hi
    else:
        if use_R: # R is numerically better when alpha is small
            from rpy2.robjects.packages import importr
            stats = importr('stats')

            hi = stats.qbeta(1 - alpha, int(k+1), int(n-k))[0]
        else:
            from scipy import stats

            hi = stats.beta.ppf(1 - alpha, k+1, n-k)
            hi = 1.0 if math.isnan(hi) else hi
    
        return hi


def bci_clopper_pearson_worst(k, n, alpha):
    return bci_clopper_pearson(k, n, alpha, two_side=False)
    
    
def estimate_bin_density(k, n, alpha):
    lo, hi = bci_clopper_pearson(k, n, alpha)
    return lo, hi


def binedges_equalmass(x, n_bins):
    n = len(x)
    return np.interp(np.linspace(0, n, n_bins + 1),
                     np.arange(n),
                     np.sort(x))

def estimate_iw_max(mdl_iw, ld, device, alpha=0.0):
    iw_list = []
    for x, y in ld:
        x = x.to(device)
        with tc.no_grad():
            w = mdl_iw(x, y)
        iw_list.append(w)
    iw_list = tc.cat(iw_list)

    iw_sorted = iw_list.sort()[0]
    iw_max = iw_sorted[math.ceil(len(iw_list)*(1.0 - alpha))-1]
    return iw_max


def find_bin_edges_equal_mass_src(ld_train, n_bins, mdl, device):
    ## compute iw using training set
    w_list_train = []
    for x, _ in ld_train:
        x = x.to(device)
        with tc.no_grad():
            w = mdl(x)
        w_list_train.append(w)
    w_list_train = tc.cat(w_list_train).cpu().detach().numpy()

    bin_edges = binedges_equalmass(w_list_train, n_bins)
    bin_edges[0] = 0.0
    bin_edges[-1] = np.inf
    return bin_edges

