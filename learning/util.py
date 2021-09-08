import sys, os
import numpy as np
from scipy import stats
import math

import torch as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.

    https://github.com/got-10k/toolkit
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).

    https://github.com/got-10k/toolkit
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def average_auc(bb_gt, bb_pred, thres=np.arange(0.0, 1.0+1e-32, 0.01)):
    ious = rect_iou(bb_gt, bb_pred)
    success_rate = []
    for t in thres:
        success_rate.append(np.mean(ious >= t))
    return np.array(success_rate).mean()



def plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, fn, fontsize=15):
    bins = np.linspace(0.0, 1.0, n_bins)
    bin_center = (bins[:-1] + bins[1:])/2.0
    conf_e, conf_t = conf_e[n_cnt>0], conf_t[n_cnt>0] 
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()
    ## acc-conf plot
    h1 = ax1.plot(conf_e, conf_t, 'ro--', label='estimated')
    h2 = ax1.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k-', label='ideal')
    ## example rate
    ax2 = ax1.twinx()
    h3 = ax2.bar(bin_center, n_cnt/np.sum(n_cnt), width=(bin_center[1]-bin_center[0])*0.75, color='b', edgecolor='k', alpha=0.5, label='ratio')
    ## beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax1.grid('on')
    ax1.set_xlabel('confidence', fontsize=fontsize)
    ax1.set_ylabel('accuracy', fontsize=fontsize)
    ax2.set_ylabel('example ratio', fontsize=fontsize)
    plt.title('ECE = %.2f%%'%(ece*100.0), fontsize=fontsize)
    plt.legend(handles=[h1[0], h2[0], h3], loc='upper left', fontsize=fontsize)
    fig.tight_layout()
    ## save
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()


def plot_acc_rank(corr, log_conf, fn, fontsize=15, ratio=0.01):

    ## sort
    corr = corr[np.argsort(log_conf, kind='stable')][::-1]  # conduct a stable sorting to properly handle tie
    
    n = len(corr)

    ranking = [float(i) for i in range(1, n+1)]
    corr_mean = [corr[:i].mean() for i in range(1, n+1)]

    n_trim = round(n*ratio)
    ranking = ranking[:n_trim]
    corr_mean = corr_mean[:n_trim]

    ## plot
    plt.figure(1)
    plt.clf()
    plt.plot(ranking, corr_mean, 'r--')

    # beautify
    plt.grid('on')
    plt.ylim((0.0, 1.0))
    plt.xlabel('ranking', fontsize=fontsize)
    plt.ylabel('average accuracy', fontsize=fontsize)
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()

    
def plot_acc_conf(corr, conf, fn, fontsize=15):

    conf_rng = np.arange(0.0, 1.0, 0.01)
    corr_mean = np.array([corr[conf>=c].mean() for c in conf_rng])
    n_cnt = np.array([np.sum(conf>=c) for c in conf_rng])

    ## plot
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()

    ## #example 
    ax2 = ax1.twinx()
    bin_center = conf_rng
    h2 = ax2.bar(bin_center, n_cnt, width=(bin_center[1]-bin_center[0]), color='b', edgecolor=None, alpha=0.3, label='#examples')

    ## curve
    h1 = ax1.plot(conf_rng, corr_mean, 'r--', label='conditional accuracy')

    # beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))

    ax1.grid('on')
    ax1.set_xlabel('confidence threshold', fontsize=fontsize)
    ax1.set_ylabel('conditional accuracy', fontsize=fontsize)
    ax2.set_ylabel('#examples', fontsize=fontsize)
    plt.legend(handles=[h2, h1[0]], fontsize=fontsize, loc='lower left')
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()
    
    
def ECE(ph, yh, y, n_bins=15, overconf=False, rel_diag_fn=None):
    assert(len(ph) == len(y))
    n = len(y)
    bins = np.linspace(0.0, 1.0, n_bins)
    conf_e = np.zeros(len(bins)-1)
    conf_t = np.zeros(len(bins)-1)
    n_cnt = np.zeros(len(bins)-1)
    
    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        idx = (ph>=l)&(ph<=u) if i==(n_bins-2) else (ph>=l)&(ph<u)
        if np.sum(idx) == 0:
            continue
        ph_i, yh_i, y_i = ph[idx], yh[idx], y[idx]
        ## compute (estimated) true confidence
        conf_t[i] = np.mean((yh_i == y_i).astype(np.float32))
        ## compute estimated confidence
        conf_e[i] = np.mean(ph_i)
        ## count the examples in the bin
        n_cnt[i] = np.sum(idx).astype(np.float32)
        
    ## expected calibration error
    ece = np.sum(np.abs(conf_e - conf_t)*n_cnt/n)
    if overconf:
        ece_oc = np.sum(np.maximum(0.0, conf_e - conf_t)*n_cnt/n)
        
    ## plot a reliability diagram
    if rel_diag_fn is not None:
        plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, rel_diag_fn)

    if overconf:
        return ece, ece_oc
    else:
        return ece
    


def compute_ece(mdl, ld, device):
    ## compute calibration error
    y_list, yh_list, ph_list = [], [], []
    mdl = mdl.to(device)
    for x, y in ld:
        x, y = x.to(device), y.to(device)
        with tc.no_grad():
            out = mdl(x)
        ph = out['ph_cal'] if 'ph_cal' in out else out['ph_top']
        yh = out['yh_cal'] if 'yh_cal' in out else out['yh_top']
        y_list.append(y.cpu())
        yh_list.append(yh.cpu())
        ph_list.append(ph.cpu())
    y_list, yh_list, ph_list = tc.cat(y_list), tc.cat(yh_list), tc.cat(ph_list)
    ece = ECE(ph_list.numpy(), yh_list.numpy(), y_list.numpy())
    return ece


def compute_error_per_label(ld, mdl, loss_fn, device):
    loss_vec, y_vec = [], []
    with tc.no_grad():
        for x, y in ld:
            loss_dict = loss_fn(x, y, mdl, reduction='none', device=device)
            loss_vec.append(loss_dict['loss'])
            y_vec.append(y)
    loss_vec, y_vec = tc.cat(loss_vec), tc.cat(y_vec)
    error_label = []
    for y in set(y_vec.tolist()):
        error_label.append(loss_vec[y_vec==y].mean().item())
    return error_label
        

def to_device(x, device):
    if tc.is_tensor(x):
        x = x.to(device)  
    elif isinstance(x, dict):
        x = {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
    elif isinstance(x, tuple):
        x = (to_device(x_i, device) for x_i in x)
    else:
        raise NotImplementedError
    return x


# def bci_clopper_pearson(k, n, alpha):
#     lo = stats.beta.ppf(alpha/2, k, n-k+1)
#     hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
#     lo = 0.0 if math.isnan(lo) else lo
#     hi = 1.0 if math.isnan(hi) else hi
    
#     return lo, hi

# def estimate_bin_density(k, n, alpha):
#     lo, hi = bci_clopper_pearson(k, n, alpha)
#     return lo, hi

    
