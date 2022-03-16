import os, sys
import time
import warnings
import pickle

from .util import *
#from learning import *
#from uncertainty import *

import learning
import uncertainty

class IWCalibrator(learning.ClsLearner):
    def __init__(self, mdl, params=None, name_postfix='iwcal'):
        super().__init__(mdl, params, name_postfix)
        
    
    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        raise NotImplementedError
        t_start = time.time()
        mdl = self.mdl if mdl is None else mdl

        ## draw a figure
        plot_wh_w_wrapper(ld, mdl, device=self.params.device,
                          fn=os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', f'plot_wh_w_{self.name_postfix}'))

# ##TODO: remove
# def create_domain_bins(iw, dom_label, n_min, n_max):

#     n_src = (dom_label==1).sum()
#     n_tar = (dom_label==0).sum()
#     iw_src = np.sort(iw[dom_label==1])
#     iw_tar = np.sort(iw[dom_label==0])

#     bin_edges = [0.0]

#     ## each bin contains at least n_min number of source "and" target
#     while True:
#         #print(len(iw_src), len(iw_tar))
#         if len(iw_src) < n_min or len(iw_tar) < n_min:
#             #bin_edges[-1] = max(iw_src.max(), iw_tar.max())
#             break
#         iw_src_i_min = iw_src[n_min-1]
#         iw_tar_i_min = iw_tar[n_min-1]
#         edge = max(iw_src_i_min, iw_tar_i_min)
#         bin_edges.append(edge)
        
#         iw_src = iw_src[iw_src>=bin_edges[-1]]
#         iw_tar = iw_tar[iw_tar>=bin_edges[-1]]
        
#     ## each bin contains at least n_min number of source "or" target
#     while True:
#         assert(len(iw_src) < n_min or len(iw_tar) < n_min)
        
#         #print(len(iw_src), len(iw_tar))
#         if len(iw_src) < n_min and len(iw_tar) < n_min:
#             if len(iw_src) == 0 and len(iw_tar) > 0:
#                 bin_edges[-1] = iw_tar.max()
#             elif len(iw_src) > 0 and len(iw_tar) == 0:
#                 bin_edges[-1] = iw_src.max()
#             elif len(iw_src) > 0 and len(iw_tar) > 0:
#                 bin_edges[-1] = max(iw_src.max(), iw_tar.max())
#             else:
#                 pass
#             break

#         if len(iw_src) < n_min:
#             edge = iw_tar[n_min-1]
#         elif len(iw_tar) < n_min:
#             edge = iw_src[n_min-1]
            
#         bin_edges.append(edge)
        
#         iw_src = iw_src[iw_src>=bin_edges[-1]]
#         iw_tar = iw_tar[iw_tar>=bin_edges[-1]]


    
#     # while True:
#     #     print(len(iw_src), len(iw_tar))
#     #     if len(iw_src) < n_min or len(iw_tar) < n_min:
#     #         bin_edges[-1] = max(iw_src.max(), iw_tar.max())
#     #         break
#     #     iw_src_i_min = iw_src[n_min-1]
#     #     iw_tar_i_min = iw_tar[n_min-1]
#     #     edge = max(iw_src_i_min, iw_tar_i_min)
#     #     bin_edges.append(edge)
        
#     #     iw_src = iw_src[iw_src>=bin_edges[-1]]
#     #     iw_tar = iw_tar[iw_tar>=bin_edges[-1]]

        
#     # while True:
#     #     print(len(iw_src), len(iw_tar))
#     #     if len(iw_src) < n_min or len(iw_tar) < n_min or len(iw_src) < n_max or len(iw_tar) < n_max:
#     #         bin_edges[-1] = max(iw_src.max(), iw_tar.max())
#     #         break
#     #     iw_src_i_min = iw_src[n_min-1]
#     #     iw_tar_i_min = iw_tar[n_min-1]
#     #     iw_src_i_max = iw_src[n_max-1]
#     #     iw_tar_i_max = iw_tar[n_max-1]

#     #     edge_min = max(iw_src_i_min, iw_tar_i_min)
#     #     edge_max = min(iw_src_i_max, iw_tar_i_max)
#     #     edge = min(edge_min, edge_max)
#     #     bin_edges.append(edge)
#     #     #bin_edges.append(max(iw_src_i_min, iw_tar_i_min))
        
#     #     iw_src = iw_src[iw_src>=bin_edges[-1]]
#     #     iw_tar = iw_tar[iw_tar>=bin_edges[-1]]
            
#     #     #assert(len(bin_edges)-1 == n_bins)
    
#     return bin_edges
    
    

class IWBinning(IWCalibrator):
    
    def __init__(self, mdl, params=None, name_postfix='iwcal_hist'):
        super().__init__(mdl, params, name_postfix)

        
    def _learn_histbin(self, iw, dom_label):

        n_bins, delta = self.mdl.n_bins.item(), self.mdl.delta.item()
        E = self.params.smoothness_bound
        print(f'## histogram binning with n_bins = {n_bins}, delta = {delta:e}, and E = {E}')
        bin_edges = self.params.bin_edges
        assert(len(bin_edges) == n_bins+1)
        
        # delta = self.mdl.delta.item()
        # print(f'## histogram binning with delta = {delta:e}')

        # # domain-equalized bining
        #raise NotImplementedError
        # bin_edges = create_domain_bins(iw, dom_label, self.params.n_binmass_min, self.params.n_binmass_max)
        # n_bins = len(bin_edges)-1

        # equal-mass binning        
        #bin_edges = binedges_equalmass(np.unique(iw), n_bins)
        #bin_edges = binedges_equalmass(iw, n_bins)
        # # equal-width binning
        # _, bin_edges = np.histogram(iw, bins=n_bins)

        # # equal-source-mass binning
        # n_bins = self.mdl.n_bins.item()
        # bin_edges = binedges_equalmass(iw_tarin_src, n_bins)
        
        
        # iw_max_est = bin_edges[-1]
        # bin_edges[0] = 0.0
        # bin_edges[-1] = np.inf

        print('[bin edges]', bin_edges)
        #print('[sample iw_max over source and target]', iw_max_est)

        n_src_list, n_tar_list, iw_est = [], [], []
        for i, (l, u) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == len(bin_edges)-2:
                idx = (iw>=l) & (iw<=u)
            else:
                idx = (iw>=l) & (iw<u)
            label_i = dom_label[idx]
            n_src = np.sum(label_i == 1)
            n_tar = np.sum(label_i == 0)

            print(f'bin_id = {i+1}, n_src = {n_src}, n_tar = {n_tar}')

            iw_est.append((l+u)/2.0)
            n_src_list.append(n_src)
            n_tar_list.append(n_tar)
        iw_est, n_src_list, n_tar_list = np.array(iw_est), np.array(n_src_list), np.array(n_tar_list)
        n_src_all, n_tar_all = np.sum(n_src_list), np.sum(n_tar_list)

        print('[src]', n_src_list, n_src_all)
        print('[tar]', n_tar_list, n_tar_all)
        
        ## estimate CP intervals
        itv_rate_src = [bci_clopper_pearson(k, n_src_all, delta / n_bins / 2.0) for k in n_src_list]
        itv_rate_tar = [bci_clopper_pearson(k, n_tar_all, delta / n_bins / 2.0) for k in n_tar_list]

        print('[itv_rate_src]', itv_rate_src)
        print('[itv_rate_tar]', itv_rate_tar)

        ## compute iw lower/upper/mean. Note that add a small value to avoid numerical error
        iw_lower = np.array([max(0, n_tar[0] - E)/(n_src[1] + E + 1e-16) for n_src, n_tar in zip(itv_rate_src, itv_rate_tar)]) 
        iw_upper = np.array([(n_tar[1] + E)/(max(0, n_src[0] - E) + 1e-16) for n_src, n_tar in zip(itv_rate_src, itv_rate_tar)])
        iw_mean = (iw_lower + iw_upper) / 2.0
        
        print('[lower]', iw_lower)
        print('[upper]', iw_upper)
        print('[mean]', iw_mean)
        print('[iw_max]', np.max(iw_upper))
        print()

        # ## estimate the interval of the sum of iw
        # n_h, a_h, b_h, delta_h = np.sum(dom_label==1), 0, np.max(iw_upper), delta/3.0
        # e = estimate_mean_hoeffding(None, n=n_h, a=a_h, b=b_h, delta=delta_h, ret_est_err=True)
        # assert(e <= 1)
        # itv_sum = [n_h*(1.0 - e), n_h*(1.0 + e)]
        # print(f'[interval of the sum of iw, n={n_h}, a={a_h}, b={b_h}, delta={delta_h}] ({itv_sum[0]}, {itv_sum[1]})')
        # print()
        
        # ## resale the upper/lower bound such that the uncertainty set of IW can be non-empty
        # iw_sum_lower, iw_sum_upper = 0.0, 0.0
        # for iw_src in iw[dom_label==1]:
        #     for l_bin, u_bin, l_iw, u_iw in zip(bin_edges[:-1], bin_edges[1:], iw_lower, iw_upper):
        #         if l_bin <= iw_src and iw_src < u_bin:
        #             iw_sum_lower += l_iw
        #             iw_sum_upper += u_iw
        #             break

        # print(f'[interval of the sum of iw by binning] ({iw_sum_lower}, {iw_sum_upper})')
        
        ## save
        self.mdl.n_bins.data = tc.tensor(n_bins, device=self.params.device)
        self.mdl.bins.data = tc.tensor(bin_edges, device=self.params.device).float()
        self.mdl.mean.data = tc.tensor(iw_mean, device=self.params.device).float()
        self.mdl.lower.data = tc.tensor(iw_lower, device=self.params.device).float()
        self.mdl.upper.data = tc.tensor(iw_upper, device=self.params.device).float()
        #self.mdl.itv_sum.data = tc.tensor(itv_sum, device=self.params.device).float()
        self.mdl.iw_max = tc.tensor(np.max(iw_upper), device=self.params.device).float()
        

    def train(self, ld_val, ld_test=None):
        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return
        
        ## load confidence prediction and correctness on label prediction
        mdl = self.mdl.mdl_iw.eval().to(self.params.device)

        ## compute iw using calibration set
        w_list, dom_label_list = [], []
        for ld, dom_label in zip(ld_val, [1, 0]):
            for x, _ in ld:
                x = x.to(self.params.device)
                with tc.no_grad():
                    w = mdl(x)
                w_list.append(w)
                dom_label_list.append((tc.ones(w.shape[0])*dom_label).long())
        w_list, dom_label_list = tc.cat(w_list), tc.cat(dom_label_list)
        w_list, dom_label_list = w_list.cpu().detach().numpy(), dom_label_list.cpu().detach().numpy()

        # ## compute iw using training set
        # w_list_train_src = []
        # for x, _ in ld_train:
        #     x = x.to(self.params.device)
        #     with tc.no_grad():
        #         w = mdl(x)
        #     w_list_train_src.append(w)
        # w_list_train_src = tc.cat(w_list_train_src).cpu().detach().numpy()


        ## estimate interval for each bins
        self._learn_histbin(w_list, dom_label_list)
        self._save_model()

        ## save the final model
        self._train_end(ld_val, ld_test)
        

class IWBinningNaive(IWBinning):
    
    def __init__(self, mdl, params=None, name_postfix='iwcal_hist_naive'):
        super().__init__(mdl, params, name_postfix)


    def _learn_histbin(self, iw, dom_label):

        n_bins = self.mdl.n_bins.item()
        print(f'## naive histogram binning with n_bins = {n_bins}')
        bin_edges = self.params.bin_edges
        assert(len(bin_edges) == n_bins+1)
        print('[bin edges]', bin_edges)

        n_src_list, n_tar_list, iw_est = [], [], []
        for i, (l, u) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == len(bin_edges)-2:
                idx = (iw>=l) & (iw<=u)
            else:
                idx = (iw>=l) & (iw<u)
            label_i = dom_label[idx]
            n_src = np.sum(label_i == 1)
            n_tar = np.sum(label_i == 0)

            print(f'bin_id = {i+1}, n_src = {n_src}, n_tar = {n_tar}')

            iw_est.append((l+u)/2.0)
            n_src_list.append(n_src)
            n_tar_list.append(n_tar)
        iw_est, n_src_list, n_tar_list = np.array(iw_est), np.array(n_src_list), np.array(n_tar_list)
        n_src_all, n_tar_all = np.sum(n_src_list), np.sum(n_tar_list)

        print('[src]', n_src_list, n_src_all)
        print('[tar]', n_tar_list, n_tar_all)
        
        ## compute iw mean. Note that add a small value to avoid numerical error
        iw_mean = np.array([n_tar/(n_src + 1e-16) for n_src, n_tar in zip(n_src_list, n_tar_list)]) 
        iw_max = np.max(iw_mean)
        print('[mean]', iw_mean)
        print('[iw_max]', iw_max)
        print()

        
        ## save
        self.mdl.n_bins.data = tc.tensor(n_bins, device=self.params.device)
        self.mdl.bins.data = tc.tensor(bin_edges, device=self.params.device).float()
        self.mdl.mean.data = tc.tensor(iw_mean, device=self.params.device).float()
        self.mdl.iw_max = tc.tensor(iw_max, device=self.params.device).float()

        
# class IWBinningPerDomain(IWBinning):
    
#     def __init__(self, mdl, params=None, name_postfix='iwcal_hist_perdomain'):
#         super().__init__(mdl, params, name_postfix)

        
#     def _learn_histbin_per_domain(self, iw, n_bins, delta):

#         # equal-mass binning
#         #bin_edges = binedges_equalmass(np.unique(iw), n_bins)
#         #iw = iw + np.random.rand(*iw.shape)*1e-9 #break ties arbitrarily

#         bin_edges = binedges_equalmass(iw, n_bins)
        
#         print('[bin edges]', bin_edges)

#         n_list = []
#         for i, (l, u) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
#             if i == len(bin_edges)-2:
#                 idx = (iw>=l) & (iw<=u)
#             else:
#                 idx = (iw>=l) & (iw<u)
#             n_i = np.sum(idx)
#             n_list.append(n_i)
#         n_list = np.array(n_list)
#         n_all = np.sum(n_list)

#         # print('[src]', n_src_list, n_src_all)
#         # print('[tar]', n_tar_list, n_tar_all)
        
#         ## estimate CP intervals
#         n_itv = [bci_clopper_pearson(k, n_all, delta / n_bins) for k in n_list]

#         n_lower = [i[0] for i in n_itv]
#         n_upper = [i[1] for i in n_itv]
#         n_mean = [k/n_all for k in n_list]
        
#         print('[lower]', n_lower)
#         print('[upper]', n_upper)
#         print('[mean]', n_mean)

#         return bin_edges, n_lower, n_upper, n_mean

    
#     def _learn_max_iw(self, bins_src, bins_tar, n_lower_src, n_upper_src, n_lower_tar, n_upper_tar):
#         iw_max = []
#         for l_src, u_src, n_l_src, n_u_src in zip(bins_src[:-1], bins_src[1:], n_lower_src, n_upper_src):
#             for l_tar, u_tar, n_l_tar, n_u_tar in zip(bins_tar[:-1], bins_tar[1:], n_lower_tar, n_upper_tar):
#                 # overlap?
#                 if (u_src > l_tar) and (l_src < u_tar):
#                     iw_max_i = n_u_tar / n_l_src
#                     iw_max.append(iw_max_i)
                    
#                     #print(f'[{l_src}, {u_src}], [{l_tar}, {u_tar}]')
#         iw_max = np.max(iw_max)
#         print(f'## iw_max = {iw_max}')
#         return iw_max
    
        
#     def _learn_histbin(self, iw, dom_label):
#         n_bins, delta = self.mdl.n_bins.item(), self.mdl.delta.item()
#         print(f'## histogram binning with n_bins = {n_bins} and delta = {delta:e}')

#         ## estimate per domain rate
#         bins_edges_src, rate_lower_src, rate_upper_src, rate_mean_src = self._learn_histbin_per_domain(iw[dom_label==1], n_bins, delta/3.0)
#         bins_edges_tar, rate_lower_tar, rate_upper_tar, rate_mean_tar = self._learn_histbin_per_domain(iw[dom_label==0], n_bins, delta/3.0)

#         ## estimate the maximum IW
#         iw_max = self._learn_max_iw(bins_edges_src, bins_edges_tar, rate_lower_src, rate_upper_src, rate_lower_tar, rate_upper_tar)
#         # update the first and last bin edge
#         bins_edges_src[0], bins_edges_tar[0] = 0.0, 0.0
#         bins_edges_src[-1], bins_edges_tar[-1] = np.inf, np.inf

#         ## estimate the interval of the sum of iw
#         warnings.warn('estimate the interval of the sum of iw')


#         ## save
#         self.mdl.bins_src.data = tc.tensor(bins_edges_src, device=self.params.device).float()
#         self.mdl.rate_mean_src.data = tc.tensor(rate_mean_src, device=self.params.device).float()
#         self.mdl.rate_lower_src.data = tc.tensor(rate_lower_src, device=self.params.device).float()
#         self.mdl.rate_upper_src.data = tc.tensor(rate_upper_src, device=self.params.device).float()
        
#         self.mdl.bins_tar.data = tc.tensor(bins_edges_tar, device=self.params.device).float()
#         self.mdl.rate_mean_tar.data = tc.tensor(rate_mean_tar, device=self.params.device).float()
#         self.mdl.rate_lower_tar.data = tc.tensor(rate_lower_tar, device=self.params.device).float()
#         self.mdl.rate_upper_tar.data = tc.tensor(rate_upper_tar, device=self.params.device).float()


def load_model_source_disc(args, mdl):

    ## init the source discriminator model
    print("## init models for iw: %s"%(args.model.sd))
    mdl_sd = model.SourceDisc(getattr(model, args.model.sd)(args.model.feat_dim, 2), mdl)
    assert(args.model_sd.path_pretrained is not None)
    print(f'## load a pretrained source discriminator at {args.model_sd.path_pretrained}')
    mdl_sd.load_state_dict(tc.load(args.model_sd.path_pretrained, map_location=tc.device('cpu')), strict=False)
    mdl_sd.eval()
    print()

    return mdl_sd


def init_iw_model(args, mdl_sd):
    ## init the IW model
    mdl_cal = model.NoCal(mdl_sd, cal_target=args.cal_sd.cal_target)    
    mdl_iw = model.IW(mdl_cal, bound_type='mean') ## choose the uncalibrated iw
    mdl_iw.eval()

    return mdl_iw


def est_iw_srcdisc(args, mdl, ds_dom):
    
    ## load the pretrained source discriminator
    mdl_sd = load_model_source_disc(args, mdl)

    ## eval the source discriminator
    l = learning.ClsLearner(mdl_sd, args.train_sd, name_postfix='srcdisc')
    #print("## test...(skip)")
    #l.test(ds_dom.test, ld_name='domain dataset', verbose=True)
    print()

    ## init an IW model
    mdl_iw = init_iw_model(args, mdl_sd)
    
    ## estimate the maximum IW
    fn_iw_max = os.path.join(os.path.dirname(args.model_sd.path_pretrained), f'iw_max_src_{args.data.src}_tar_{args.data.tar}_alpha_0.0.pk')
    if os.path.exists(fn_iw_max):
        iw_max = pickle.load(open(fn_iw_max, 'rb'))
        print(f'## iw_max loaded from {fn_iw_max}')
    else:
        iw_max = uncertainty.estimate_iw_max(mdl_iw, ds_dom.train, args.device, alpha=0.0)
        pickle.dump(iw_max, open(fn_iw_max, 'wb'))
    mdl_iw.iw_max.data = tc.tensor(iw_max.item(), device=mdl_iw.iw_max.data.device)
    print("## iw_max over train = %f (before cal)"%(iw_max))
    print()

    return args, mdl_iw


def est_iw_temp(args, mdl, ds_dom):
    ## load the pretrained source discriminator
    mdl_sd = load_model_source_disc(args, mdl)

    ## calibrate the source discriminator model via temperature scaling
    mdl_sd = model.Temp(mdl_sd)
    l = uncertainty.TempScalingLearner(mdl_sd, args.cal_sd, name_postfix='srcdisc_temp')
    print("## test before calibration...")
    l.test(ds_dom.test, ld_name='domain data', verbose=True)
    print("## calibrate...")
    l.train(ds_dom.val, ds_dom.val)
    print("## test after calibration...")
    l.test(ds_dom.test, ld_name='domain data', verbose=True)
    print()

    ## init an IW model
    mdl_iw = init_iw_model(args, mdl_sd)

    ## estimate the maximum IW before calibration
    # always recompute the max since the max can be changed as a calibration set changes
    iw_max = uncertainty.estimate_iw_max(mdl_iw, ds_dom.train, args.device)
    mdl_iw.iw_max.data = tc.tensor(iw_max.item(), device=mdl_iw.iw_max.data.device)
    print("## iw_max over train = %f (before cal)"%(iw_max))
    print()

    return args, mdl_iw


def est_iw_bin_mean(args, mdl, ds_src, ds_tar):
    
    ## load the pretrained source discriminator
    mdl_sd = load_model_source_disc(args, mdl)

    ## eval the source discriminator
    l = learning.ClsLearner(mdl_sd, args.train_sd, name_postfix='srcdisc')
    #print("## test...(skip)")
    #l.test(ds_dom.test, ld_name='domain dataset', verbose=True)
    print()

    ## init an IW model
    mdl_iw = init_iw_model(args, mdl_sd)

    ## calibrate the IW model
    print("## calibrate IW...")

    fn_bin_edges = os.path.join(os.path.dirname(args.model_sd.path_pretrained), f'bin_edges_n_bins_{args.model_iwcal.n_bins}_src_equal_mass.pk')

    if os.path.exists(fn_bin_edges):
        bin_edges = pickle.load(open(fn_bin_edges, 'rb'))
        print(f'## bin_edges loaded from {fn_bin_edges}:', bin_edges)
        assert(len(bin_edges) == args.model_iwcal.n_bins+1)
    else:
        bin_edges = uncertainty.find_bin_edges_equal_mass_src(ds_src.train, args.model_iwcal.n_bins, mdl_iw, args.device)
        pickle.dump(bin_edges, open(fn_bin_edges, 'wb'))
        print(f'## bin_edges saved to {fn_bin_edges}:', bin_edges)
    args.cal_iw.bin_edges = bin_edges

    mdl_iw = model.IWCal(mdl_iw, n_bins=args.model_iwcal.n_bins, delta=args.model_iwcal.delta)
    l = uncertainty.IWBinningNaive(mdl_iw, args.cal_iw, name_postfix='iwcal_hist_naive')
    l.train([ds_src.val, ds_tar.val])
    #l.test([ds_src.test, ds_tar.test], ld_name='domain dataset', verbose=True)
    print()

    return args, mdl_iw


def est_iw_bin_interval(args, mdl, ds_src, ds_tar):

    ## load the pretrained source discriminator
    mdl_sd = load_model_source_disc(args, mdl)

    ## eval the source discriminator
    l = learning.ClsLearner(mdl_sd, args.train_sd, name_postfix='srcdisc')
    #print("## test...(skip)")
    #l.test(ds_dom.test, ld_name='domain dataset', verbose=True)
    print()

    ## init an IW model
    mdl_iw = init_iw_model(args, mdl_sd)

    ## calibrate the IW model
    print("## calibrate IWs...")
    fn_bin_edges = os.path.join(os.path.dirname(args.model_sd.path_pretrained), f'bin_edges_n_bins_{args.model_iwcal.n_bins}_src_equal_mass.pk')

    if os.path.exists(fn_bin_edges):
        bin_edges = pickle.load(open(fn_bin_edges, 'rb'))
        print(f'## bin_edges is loaded from {fn_bin_edges}:', bin_edges)
        assert(len(bin_edges) == args.model_iwcal.n_bins+1)
    else:
        bin_edges = uncertainty.find_bin_edges_equal_mass_src(ds_src.train, args.model_iwcal.n_bins, mdl_iw, args.device)
        pickle.dump(bin_edges, open(fn_bin_edges, 'wb'))
        print(f'## bin_edges is saved to {fn_bin_edges}:', bin_edges)
    args.cal_iw.bin_edges = bin_edges

    mdl_iw = model.IWCal(mdl_iw, n_bins=args.model_iwcal.n_bins, delta=args.model_iwcal.delta)
    l = uncertainty.IWBinning(mdl_iw, args.cal_iw, name_postfix='iwcal_hist')
    l.train([ds_src.val, ds_tar.val])
    #l.test([ds_src.test, ds_tar.test], ld_name='domain dataset', verbose=True)
    print()

    return args, mdl_iw
