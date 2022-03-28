import os, sys
import argparse
import warnings
import numpy as np
import math
import pickle

import torch as tc

import util
import data
import model
import learning
import uncertainty


    
def run(args):

    ## init datasets
    print("## init source datasets: %s"%(args.data.src))
    ds_src = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        batch_size=args.data.batch_size,
        image_size=None if args.data.img_size is None else args.data.img_size[1],
        dim=args.data.dim,
        train_rnd=True, val_rnd=True, test_rnd=False, 
        #train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
        #aug_types=args.data.aug_src,
        color=True if args.data.img_size is not None and args.data.img_size[0]==3 else False,
        sample_size={'train': args.data.n_train_src, 'val': args.data.n_val_src, 'test': args.data.n_test_src},
        seed=args.data.seed,
        num_workers=args.data.n_workers,
        load_feat=args.data.load_feat,
        normalize=not args.model.normalize,
    )
    print()
    
    print("## init target datasets: %s"%(args.data.tar))
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        image_size=None if args.data.img_size is None else args.data.img_size[1],
        dim=args.data.dim,
        train_rnd=True, val_rnd=True, test_rnd=False, 
        #train_aug=args.data.aug_tar is not None, val_aug=args.data.aug_tar is not None, test_aug=args.data.aug_tar is not None,
        #aug_types=args.data.aug_tar,
        color=True if args.data.img_size is not None and args.data.img_size[0]==3 else False,
        sample_size={'train': args.data.n_train_tar, 'val': args.data.n_val_tar, 'test': args.data.n_test_tar},
        seed=args.data.seed,
        num_workers=args.data.n_workers,
        load_feat=args.data.load_feat,
        normalize=not args.model.normalize,
    )
    print()
    
    print("## init domain datasets: src = %s, tar = %s"%(args.data.src, args.data.tar))
    ds_dom = data.DomainData(ds_src, ds_tar)
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))    
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=args.data.dim[0], n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained)    
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    else:
        raise NotImplementedError
    
    if args.model.normalize: # active for adversarial examples
        print('## init an image normalizer as part of model')  
        mdl = model.ExampleNormalizer(mdl)
        
    if args.data.load_feat:
        print("## init models: %s"%(args.model.base_feat))
        mdl = getattr(model, args.model.base_feat)(mdl)
    print()
    
    ## learning
    assert(args.model.path_pretrained is not None)
    l = learning.ClsLearner(mdl, args.train)
    print("## test...(skip)")
    #l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    print()

    # ## IW
    # ## use true iw
    # if args.model.iw_true:
    #     ## load true IW
    #     assert('Normal' in args.data.src and 'Normal' in args.data.tar)
    #     if args.data.src == 'FatNormal':
    #         mu = tc.zeros(args.data.dim)
    #         sig = tc.ones(args.data.dim) * 1e-1
    #         sig[0] = 5.0
    #         p_params = {'mu': mu, 'sig': sig}
    #     elif args.data.src == 'ThinNormal':
    #         mu = tc.zeros(args.data.dim)
    #         sig = tc.ones(args.data.dim) * 1e-1
    #         sig[0] = 1.0
    #         p_params = {'mu': mu, 'sig': sig}
    #     else:
    #         raise NotImplementedError

    #     if args.data.tar == 'FatNormal':
    #         mu = tc.zeros(args.data.dim)
    #         sig = tc.ones(args.data.dim) * 1e-1
    #         sig[0] = 5.0
    #         q_params = {'mu': mu, 'sig': sig}
    #     elif args.data.tar == 'ThinNormal':
    #         mu = tc.zeros(args.data.dim)
    #         sig = tc.ones(args.data.dim) * 1e-1 
    #         sig[0] = 1.0
    #         q_params = {'mu': mu, 'sig': sig}
    #     else:
    #         raise NotImplementedError
        
    #     mdl_iw = IWTwoNormals(p_params=p_params, q_params=q_params, device=args.device)
    #     print(f"## true iw_max = {mdl_iw.iw_max}")
    #     print()
    # else:
    #     pass
    #     # ## estimate IW

    #     # ## init the source discriminator model
    #     # print("## init models for iw: %s"%(args.model.sd))
    #     # mdl_sd = model.SourceDisc(getattr(model, args.model.sd)(args.model.feat_dim, 2), mdl)
    #     # assert(args.model_sd.path_pretrained is not None)
    #     # print(f'## load a pretrained source discriminator at {args.model_sd.path_pretrained}')
    #     # mdl_sd.load_state_dict(tc.load(args.model_sd.path_pretrained, map_location=tc.device('cpu')), strict=False)
    #     # mdl_sd.eval()
    #     # print()

    #     # ## calibrate the source discriminator model
    #     # if args.train_predset.method == 'pac_predset_temp_rejection':
    #     #     mdl_sd = model.Temp(mdl_sd)
    #     #     l = uncertainty.TempScalingLearner(mdl_sd, args.cal_sd, name_postfix='srcdisc_temp')
    #     #     print("## test before calibration...")
    #     #     l.test(ds_dom.test, ld_name='domain data', verbose=True)
    #     #     print("## calibrate...")
    #     #     l.train(ds_dom.val, ds_dom.val)
    #     #     print("## test after calibration...")
    #     #     l.test(ds_dom.test, ld_name='domain data', verbose=True)
    #     #     print()

    #     # ## eval IW
    #     # l = learning.ClsLearner(mdl_sd, args.train_sd, name_postfix='srcdisc')
    #     # print("## test...(skip)")
    #     # #l.test(ds_dom.test, ld_name='domain dataset', verbose=True)
    #     # print()

    #     # ## init the IW model
    #     # mdl_cal = model.NoCal(mdl_sd, cal_target=args.cal_sd.cal_target)    
    #     # mdl_iw = model.IW(mdl_cal, bound_type='mean') ## choose the uncalibrated iw
    #     # mdl_iw.eval()

    #     # ## estimate the maximum IW before calibration
    #     # if args.train_predset.method == 'pac_predset_temp_rejection':
    #     #     # always recompute max since max can be changed as a calibration set changes
    #     #     iw_max = uncertainty.estimate_iw_max(mdl_iw, ds_dom.train, args.device)
    #     # else:            
    #     #     fn_iw_max = os.path.join(os.path.dirname(args.model_sd.path_pretrained), f'iw_max_src_{args.data.src}_tar_{args.data.tar}_alpha_0.0.pk')
    #     #     if os.path.exists(fn_iw_max):
    #     #         iw_max = pickle.load(open(fn_iw_max, 'rb'))
    #     #         print(f'## iw_max loaded from {fn_iw_max}')
    #     #     else:
    #     #         iw_max = uncertainty.estimate_iw_max(mdl_iw, ds_dom.train, args.device, alpha=0.0)
    #     #         pickle.dump(iw_max, open(fn_iw_max, 'wb'))
    #     # mdl_iw.iw_max.data = tc.tensor(iw_max.item(), device=mdl_iw.iw_max.data.device)
    #     # print("## iw_max over train = %f (before cal)"%(iw_max))
    #     # print()


    #     # ## calibrate the IW model
    #     # if args.train_predset.method == 'pac_predset_worst_rejection' or args.train_predset.method == 'pac_predset_mean_rejection':
    #     #     print("## calibrate IW...")

    #     #     fn_bin_edges = os.path.join(os.path.dirname(args.model_sd.path_pretrained), f'bin_edges_n_bins_{args.model_iwcal.n_bins}_src_equal_mass.pk')

    #     #     if os.path.exists(fn_bin_edges):
    #     #         bin_edges = pickle.load(open(fn_bin_edges, 'rb'))
    #     #         print(f'## bin_edges loaded from {fn_bin_edges}:', bin_edges)
    #     #         assert(len(bin_edges) == args.model_iwcal.n_bins+1)
    #     #     else:
    #     #         bin_edges = uncertainty.find_bin_edges_equal_mass_src(ds_src.train, args.model_iwcal.n_bins, mdl_iw, args.device)
    #     #         pickle.dump(bin_edges, open(fn_bin_edges, 'wb'))
    #     #         print(f'## bin_edges saved to {fn_bin_edges}:', bin_edges)

    #     #     args.cal_iw.bin_edges = bin_edges

    #     #     ## calibrate IWs
    #     #     if args.train_predset.method == 'pac_predset_worst_rejection':

    #     #         mdl_iw = model.IWCal(mdl_iw, n_bins=args.model_iwcal.n_bins, delta=args.model_iwcal.delta)
    #     #         l = uncertainty.IWBinning(mdl_iw, args.cal_iw, name_postfix='iwcal_hist')
    #     #         l.train([ds_src.val, ds_tar.val])
    #     #         #l.test([ds_src.test, ds_tar.test], ld_name='domain dataset', verbose=True)

    #     #     elif args.train_predset.method == 'pac_predset_mean_rejection':

    #     #         mdl_iw = model.IWCal(mdl_iw, n_bins=args.model_iwcal.n_bins, delta=args.model_iwcal.delta)
    #     #         l = uncertainty.IWBinningNaive(mdl_iw, args.cal_iw, name_postfix='iwcal_hist_naive')
    #     #         l.train([ds_src.val, ds_tar.val])

    #     #     print()


    ## construct and evaluate a prediction set
    if args.train_predset.method == 'pac_predset':
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor(mdl_predset, args.train_predset, model_iw=None)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'pac_predset_CP':
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_CP(mdl_predset, args.train_predset, model_iw=None)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'split_cp':
        # construct a prediction set
        mdl_predset = model.SplitCPCls(mdl, args.model_predset.alpha, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.SplitCPConstructor(mdl_predset, args.train_predset)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'weighted_split_cp':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_srcdisc(args, mdl, ds_dom)
        # construct a prediction set
        mdl_predset = model.WeightedSplitCPCls(mdl, mdl_iw, args.model_predset.alpha, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.WeightedSplitCPConstructor(mdl_predset, params=args.train_predset, mdl_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'pac_predset_maxiw':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_srcdisc(args, mdl, ds_dom)
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_maxiw(mdl_predset, args.train_predset, model_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'pac_predset_rejection':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_srcdisc(args, mdl, ds_dom)
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
    elif args.train_predset.method == 'pac_predset_temp_rejection':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_temp(args, mdl, ds_dom)
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

    elif args.train_predset.method == 'pac_predset_mean_rejection':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_bin_mean(args, mdl, ds_src, ds_tar)
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    
    elif args.train_predset.method == 'pac_predset_worst_rejection':
        # estimate IWs
        if args.model.iw_true:
            mdl_iw = uncertainty.get_two_gaussian_true_iw(args)
        else:
            args, mdl_iw = uncertainty.est_iw_bin_interval(args, mdl, ds_src, ds_tar)
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_worst_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)
        l.train(ds_src.val)
        # evaluate
        l.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    else:
        raise NotImplementedError


    
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--train_iw', action='store_true')
    #parser.add_argument('--estimate', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=100)
    parser.add_argument('--data.n_workers', type=int, default=0)
    parser.add_argument('--data.src', type=str, required=True)
    parser.add_argument('--data.tar', type=str, required=True)
    parser.add_argument('--data.n_labels', type=int)
    parser.add_argument('--data.img_size', type=int, nargs=3)
    parser.add_argument('--data.dim', type=int, nargs='*')
    parser.add_argument('--data.aug_src', type=str, nargs='*')
    parser.add_argument('--data.aug_tar', type=str, nargs='*')
    parser.add_argument('--data.n_train_src', type=int)
    parser.add_argument('--data.n_train_tar', type=int)
    parser.add_argument('--data.n_val_src', type=int)
    parser.add_argument('--data.n_val_tar', type=int)
    parser.add_argument('--data.n_test_src', type=int)
    parser.add_argument('--data.n_test_tar', type=int)
    parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)
    parser.add_argument('--data.load_feat', type=str)

    ## model args
    parser.add_argument('--model.base', type=str)
    parser.add_argument('--model.base_feat', type=str)
    parser.add_argument('--model.path_pretrained', type=str)
    parser.add_argument('--model.feat_dim', type=int)
    parser.add_argument('--model.cal', type=str, default='Temp')
    parser.add_argument('--model.sd', type=str, default='MidFNN')
    parser.add_argument('--model.sd_cal', type=str, default='HistBin')
    parser.add_argument('--model.normalize', action='store_true')
    parser.add_argument('--model.iw_true', action='store_true')

    parser.add_argument('--model_sd.path_pretrained', type=str)
    parser.add_argument('--model_iwcal.n_bins', type=int, default=10) ## can be changed depending on binning scheme
    parser.add_argument('--model_iwcal.delta', type=float)

    ## predset model args
    parser.add_argument('--model_predset.eps', type=float, default=0.1)
    parser.add_argument('--model_predset.alpha', type=float, default=0.1)
    parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    parser.add_argument('--model_predset.m', type=int)

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')

    ## calibration args
    parser.add_argument('--cal.method', type=str, default='HistBin')
    parser.add_argument('--cal.rerun', action='store_true')
    parser.add_argument('--cal.load_final', action='store_true')
    parser.add_argument('--cal.optimizer', type=str, default='SGD')
    parser.add_argument('--cal.n_epochs', type=int, default=100)
    parser.add_argument('--cal.lr', type=float, default=0.01)
    parser.add_argument('--cal.momentum', type=float, default=0.9)
    parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal.val_period', type=int, default=1)    

    ## train args for a source discriminator
    parser.add_argument('--train_sd.rerun', action='store_true')
    parser.add_argument('--train_sd.load_final', action='store_true')
    parser.add_argument('--train_sd.optimizer', type=str, default='SGD')
    parser.add_argument('--train_sd.n_epochs', type=int, default=100)
    parser.add_argument('--train_sd.lr', type=float, default=0.01)
    parser.add_argument('--train_sd.momentum', type=float, default=0.9)
    parser.add_argument('--train_sd.weight_decay', type=float, default=0.0)
    parser.add_argument('--train_sd.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train_sd.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train_sd.val_period', type=int, default=1)

    ## calibration args for a source discriminator
    parser.add_argument('--cal_sd.method', type=str, default='HistBin')
    parser.add_argument('--cal_sd.rerun', action='store_true')
    parser.add_argument('--cal_sd.resume', action='store_true')
    parser.add_argument('--cal_sd.load_final', action='store_true')
    ## histbin parameters
    parser.add_argument('--cal_sd.delta', type=float, default=1e-5)
    parser.add_argument('--cal_sd.estimate_rate', action='store_true')
    parser.add_argument('--cal_sd.cal_target', type=int, default=1)
    ## temp parameters
    parser.add_argument('--cal_sd.optimizer', type=str, default='SGD')
    parser.add_argument('--cal_sd.n_epochs', type=int, default=100) 
    parser.add_argument('--cal_sd.lr', type=float, default=0.01)
    parser.add_argument('--cal_sd.momentum', type=float, default=0.9)
    parser.add_argument('--cal_sd.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal_sd.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal_sd.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal_sd.val_period', type=int, default=1)    

    ## iw calibration args
    parser.add_argument('--cal_iw.method', type=str, default='HistBin')
    parser.add_argument('--cal_iw.rerun', action='store_true')
    parser.add_argument('--cal_iw.load_final', action='store_true')
    parser.add_argument('--cal_iw.smoothness_bound', type=float, default=0.001)

    ## uncertainty estimation args
    parser.add_argument('--train_predset.method', type=str, default='pac_predset')
    parser.add_argument('--train_predset.rerun', action='store_true')
    parser.add_argument('--train_predset.load_final', action='store_true')
    parser.add_argument('--train_predset.binary_search', action='store_true')
    parser.add_argument('--train_predset.bnd_type', type=str, default='direct')

    parser.add_argument('--train_predset.T_step', type=float, default=1e-7) 
    parser.add_argument('--train_predset.T_end', type=float, default=np.inf)
    parser.add_argument('--train_predset.eps_tol', type=float, default=1.5)
        
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')
    
    ## set loggers
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    ## print args
    util.print_args(args)
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    run(args)


