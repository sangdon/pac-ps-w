import os, sys
import argparse
import warnings
import numpy as np
import math
import pickle
import shutil

import torch as tc

import util
import data
import model
import learning
import uncertainty



    
def main(args):

    ## print args
    util.print_args(args)

    ## init datasets
    # print("## init source datasets: %s"%(args.data.src))
    # ds_src = getattr(data, args.data.src)(
    #     root=os.path.join('data', args.data.src.lower()),
    #     batch_size=args.data.batch_size,
    #     image_size=None if args.data.img_size is None else args.data.img_size[1],
    #     dim=args.data.dim,
    #     train_rnd=True, val_rnd=True, test_rnd=False, 
    #     train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
    #     aug_types=args.data.aug_src,
    #     color=True if args.data.img_size is not None and args.data.img_size[0]==3 else False,
    #     sample_size={'train': args.data.n_train_src, 'val': args.data.n_val_src, 'test': args.data.n_test_src},
    #     seed=args.data.seed,
    #     num_workers=args.data.n_workers,
    #     load_feat=args.data.load_feat,
    #     normalize=not args.model.normalize,
    # )
    # print()

    # fontsize = args.fontsize
    # fig_root = f'{args.snapshot_root}/figs/{args.dataset.split("_")[0]}/{args.dataset}'

    
    print("## init target datasets: %s"%(args.data.tar))
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        image_size=None if args.data.img_size is None else args.data.img_size[1],
        dim=args.data.dim,
        train_rnd=True, val_rnd=True, test_rnd=False, 
        train_aug=args.data.aug_tar is not None, val_aug=args.data.aug_tar is not None, test_aug=args.data.aug_tar is not None,
        aug_types=args.data.aug_tar,
        color=True if args.data.img_size is not None and args.data.img_size[0]==3 else False,
        sample_size={'train': args.data.n_train_tar, 'val': args.data.n_val_tar, 'test': args.data.n_test_tar},
        seed=args.data.seed,
        num_workers=args.data.n_workers,
        load_feat=args.data.load_feat,
        normalize=not args.model.normalize,
        return_path=True,
    )
    print()
    
    # print("## init domain datasets: src = %s, tar = %s"%(args.data.src, args.data.tar))
    # ds_dom = data.DomainData(ds_src, ds_tar)
    # print()

    ## init a model
    print("## init models: %s"%(args.model.base))    
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=args.data.dim[0], n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained)    
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    else:
        raise NotImplementedError
    
    if args.model.normalize:
        print('## init an image normalizer as a pre-processing model')
        mdl = model.ExampleNormalizer(mdl)
    if args.data.load_feat:
        print("## init models: %s"%(args.model.base_feat))
        mdl = getattr(model, args.model.base_feat)(mdl)
    print()

    # ## dummy iw model
    # mdl_iw = None
    
    # ## uncertainty
    # if args.estimate:
    #     if args.train_predset.method == 'pac_predset':
    #         warnings.warn('the original pac_predset might be slow if m is large')
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor(mdl_predset, args.train_predset, model_iw=None)
    #     elif args.train_predset.method == 'pac_predset_CP':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_CP(mdl_predset, args.train_predset, model_iw=None)
    #     elif args.train_predset.method == 'split_cp':
    #         mdl_predset = model.SplitCPCls(mdl, args.model_predset.alpha, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.SplitCPConstructor(mdl_predset, args.train_predset)
    #     elif args.train_predset.method == 'weighted_split_cp':
    #         mdl_predset = model.WeightedSplitCPCls(mdl, mdl_iw, args.model_predset.alpha, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.WeightedSplitCPConstructor(mdl_predset, params=args.train_predset, mdl_iw=mdl_iw)
    #     elif args.train_predset.method == 'predset_bootstrap':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_bootstrap(mdl_predset, params=args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'predset_resampling_bootstrap':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_resampling_bootstrap(mdl_predset, params=args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_worstiw':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_worstiw(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_worstbin':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_worstbin(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_worstbinopt':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_worstbinopt(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_mgf':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_MGF(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_HCP':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_HCP(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_EBCP':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_EBCP(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_wbin':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_wbin(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_resample':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_resample(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method in ['pac_predset_rejection', 'pac_predset_mean_rejection', 'pac_predset_temp_rejection'] :
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     elif args.train_predset.method == 'pac_predset_worst_rejection':
    #         mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #         l = uncertainty.PredSetConstructor_worst_rejection(mdl_predset, args.train_predset, model_iw=mdl_iw)

    #     # elif args.train_predset.method == 'pac_predset_H':
    #     #     mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #     #     l = uncertainty.PredSetConstructor_H(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     # elif args.train_predset.method == 'pac_predset_EB':
    #     #     mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.n)
    #     #     l = uncertainty.PredSetConstructor_EB(mdl_predset, args.train_predset, model_iw=mdl_iw)
    #     else:
    #         raise NotImplementedError

    # ## load ps
    # l.train(None) 
    # print('T =', (-mdl_predset.T).exp().item())

    return mdl, ds_tar
    
                

def plot(args):

    mdl, ds_tar = main(args)

    if args.dataset_id == 'domainnet':
        path_mdl_ps_iid = 'snapshots/exp_domainnet_src_DomainNetAll_tar_DomainNetPainting_da_iwcal_naive_m_50000_eps_0.1_delta_1e-5_expid_1/model_params_n_50000_eps_0.10000000149011612_delta_9.999999747378752e-06_best'
        path_mdl_ps_ours = 'snapshots/exp_domainnet_src_DomainNetAll_tar_DomainNetPainting_da_iwcal_worst_rejection_m_50000_eps_0.1_delta_1e-5_expid_1/model_params_n_50000_eps_0.10000000149011612_delta_4.999999873689376e-06_best'
    elif args.dataset_id == 'imagenetc':
        
        pass
    elif args.dataset_id == 'imagenetadvex':
        path_mdl_ps_iid = 'snapshots/exp_imagenet_src_ImageNet_tar_ImageNetAdvEx_eps_0p01_da_iwcal_naive_m_20000_eps_0.1_delta_1e-5_expid_1/model_params_n_20000_eps_0.10000000149011612_delta_9.999999747378752e-06_best'
        path_mdl_ps_ours = 'snapshots/exp_imagenet_src_ImageNet_tar_ImageNetAdvEx_eps_0p01_da_iwcal_worst_rejection_m_20000_eps_0.1_delta_1e-5_expid_1/model_params_n_20000_eps_0.10000000149011612_delta_4.999999873689376e-06_best'

    max_ps_size = 10
    n_img = 100

    f_good_ex = True
    
    ## iid
    mdl_ps_iid = model.PredSetCls(mdl, 0, 0, 0)
    mdl_ps_iid.load_state_dict(tc.load(path_mdl_ps_iid))
    print(mdl_ps_iid.T.item())
    ## ours
    mdl_ps_ours = model.PredSetCls(mdl, 0, 0, 0)
    mdl_ps_ours.load_state_dict(tc.load(path_mdl_ps_ours))
    print(mdl_ps_ours.T.item())
                                     
    ## load a meta file
    if args.dataset_id == 'domainnet':
        id_name_map = ds_tar.id_name_map
    else:
        id_name_map = ds_tar.names
    
    if f_good_ex:
        fig_root = os.path.join(args.snapshot_root, 'figs', 'ps_ex', 'tar_'+args.data.tar)
    else:
        fig_root = os.path.join(args.snapshot_root, 'figs', 'ps_ex_bad', 'tar_'+args.data.tar)

    shutil.rmtree(fig_root, ignore_errors=True)
    os.makedirs(fig_root, exist_ok=True)
    
    ## save examples with ps
    mdl_ps_iid.cuda()
    mdl_ps_ours.cuda()
    i_img = 0
    for path, x, y in ds_tar.test:
        x, y = x.cuda(), y.cuda()
        yh = mdl(x)['yh_top']
        
        ps_iid = mdl_ps_iid.set(x)
        ps_ours = mdl_ps_ours.set(x)

        ps_iid_membership = mdl_ps_iid.membership(x, y)
        ps_ours_membership = mdl_ps_ours.membership(x, y)

        for p_i, ps_iid_i, ps_ours_i, c_iid_i, c_ours_i, yh_i in zip(path, ps_iid, ps_ours, ps_iid_membership, ps_ours_membership, yh):
            if f_good_ex:
                if c_iid_i:
                    continue
                if ~c_ours_i:
                    continue
            else:
                if ~c_iid_i:
                    continue
                
            if ps_iid_i.sum() > max_ps_size or ps_ours_i.sum() > max_ps_size:
                continue

            print('----', p_i)

            y_i_str = p_i.split('/')[-2]
            yh_i_str = id_name_map[yh_i.item()]
            ps_iid_i_str = []
            for i, ps_i_flag in enumerate(ps_iid_i):
                if ps_i_flag:
                    ps_iid_i_str.append(id_name_map[i])
            ps_ours_i_str = []
            for i, ps_i_flag in enumerate(ps_ours_i):
                if ps_i_flag:
                    ps_ours_i_str.append(id_name_map[i])
                    
            fn_img_src = '.'.join(p_i.split('.')[:-1])
            fn_img_src_split = fn_img_src.split('/')
            fn_img_src_split[2] = fn_img_src_split[2].split('_')[0]
            fn_img_src = '/'.join(fn_img_src_split)
            fn_img_tar_iid = os.path.join(fig_root, fn_img_src, f"iid_y_{y_i_str}_yh_{yh_i_str}_ps_{'+'.join(ps_iid_i_str)}.png")
            fn_img_tar_ours = os.path.join(fig_root, fn_img_src, f"ours_y_{y_i_str}_yh_{yh_i_str}_ps_{'+'.join(ps_ours_i_str)}.png")

            os.makedirs(os.path.dirname(fn_img_tar_iid), exist_ok=True)

            print(f'# generate...src = {fn_img_src}, \ntar = {fn_img_tar_iid}, \ntar = {fn_img_tar_ours}\n')
            shutil.copyfile(fn_img_src, fn_img_tar_iid)
            shutil.copyfile(fn_img_src, fn_img_tar_ours)

            i_img += 1
        if i_img > n_img:
            break



def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    #parser.add_argument('--exp_name', type=str, required=True)
    #parser.add_argument('--exp_name_list', type=str, nargs=2, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--train_iw', action='store_true')
    parser.add_argument('--estimate', action='store_true')

    #parser.add_argument('--dataset', type=str, default='domainnet_src_DomainNetAll_tar_DomainNetSketch_da_iwcal')
    #parser.add_argument('--fontsize', type=int, default=20)
    #parser.add_argument('--figsize', type=float, nargs=2, default=[6.4*1.5, 4.8])
    #parser.add_argument('--gen_error_exs', action='store_true')
    parser.add_argument('--dataset_id', type=str, default='domainnet')
    
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
    parser.add_argument('--model_predset.eps', type=float, default=0.01)
    parser.add_argument('--model_predset.alpha', type=float, default=0.01)
    parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    parser.add_argument('--model_predset.n', type=int)

    # ## iw_max model args %%TODO: rm
    # parser.add_argument('--model_iw_max.eps', type=float, default=0.01)
    # parser.add_argument('--model_iw_max.delta', type=float, default=1e-3)
    # parser.add_argument('--model_iw_max.n', type=int)

    # ## train args
    # parser.add_argument('--train.rerun', action='store_true')
    # parser.add_argument('--train.load_final', action='store_true')

    # ## calibration args
    # parser.add_argument('--cal.method', type=str, default='HistBin')
    # parser.add_argument('--cal.rerun', action='store_true')
    # parser.add_argument('--cal.load_final', action='store_true')
    # parser.add_argument('--cal.optimizer', type=str, default='SGD')
    # parser.add_argument('--cal.n_epochs', type=int, default=100)
    # parser.add_argument('--cal.lr', type=float, default=0.01)
    # parser.add_argument('--cal.momentum', type=float, default=0.9)
    # parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    # parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--cal.val_period', type=int, default=1)    

    # ## train args for a source discriminator
    # parser.add_argument('--train_sd.rerun', action='store_true')
    # parser.add_argument('--train_sd.load_final', action='store_true')
    # parser.add_argument('--train_sd.optimizer', type=str, default='SGD')
    # parser.add_argument('--train_sd.n_epochs', type=int, default=100)
    # parser.add_argument('--train_sd.lr', type=float, default=0.01)
    # parser.add_argument('--train_sd.momentum', type=float, default=0.9)
    # parser.add_argument('--train_sd.weight_decay', type=float, default=0.0)
    # parser.add_argument('--train_sd.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--train_sd.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--train_sd.val_period', type=int, default=1)

    # ## calibration args for a source discriminator
    # parser.add_argument('--cal_sd.method', type=str, default='HistBin')
    # parser.add_argument('--cal_sd.rerun', action='store_true')
    # parser.add_argument('--cal_sd.resume', action='store_true')
    # parser.add_argument('--cal_sd.load_final', action='store_true')
    # ## histbin parameters
    # parser.add_argument('--cal_sd.delta', type=float, default=1e-5)
    # parser.add_argument('--cal_sd.estimate_rate', action='store_true')
    # parser.add_argument('--cal_sd.cal_target', type=int, default=1)
    # ## temp parameters
    # parser.add_argument('--cal_sd.optimizer', type=str, default='SGD')
    # parser.add_argument('--cal_sd.n_epochs', type=int, default=100) 
    # parser.add_argument('--cal_sd.lr', type=float, default=0.01)
    # parser.add_argument('--cal_sd.momentum', type=float, default=0.9)
    # parser.add_argument('--cal_sd.weight_decay', type=float, default=0.0)
    # parser.add_argument('--cal_sd.lr_decay_epoch', type=int, default=20)
    # parser.add_argument('--cal_sd.lr_decay_rate', type=float, default=0.5)
    # parser.add_argument('--cal_sd.val_period', type=int, default=1)    

    # ## iw calibration args
    # parser.add_argument('--cal_iw.method', type=str, default='HistBin')
    # parser.add_argument('--cal_iw.rerun', action='store_true')
    # parser.add_argument('--cal_iw.load_final', action='store_true')
    # parser.add_argument('--cal_iw.smoothness_bound', type=float, default=0.001)
    # # parser.add_argument('--cal_iw.n_binmass_min', type=int, default=2000)
    # # parser.add_argument('--cal_iw.n_binmass_max', type=int, default=4000)

    # # ## iw_max estimation args
    # # parser.add_argument('--train_iw_max.method', type=str, default='pac_predset_CP')
    # # parser.add_argument('--train_iw_max.rerun', action='store_true')
    # # parser.add_argument('--train_iw_max.load_final', action='store_true')
    # # parser.add_argument('--train_iw_max.binary_search', action='store_true')
    # # parser.add_argument('--train_iw_max.bnd_type', type=str, default='direct')
    # # parser.add_argument('--train_iw_max.T_step', type=float, default=1e-5)
    # # parser.add_argument('--train_iw_max.T_end', type=float, default=np.inf)
    # # parser.add_argument('--train_iw_max.eps_tol', type=float, default=1.5)
        

    # ## uncertainty estimation args
    # #parser.add_argument('--train_predset.method', type=str, default='pac_predset')
    # parser.add_argument('--train_predset.method_list', type=str, nargs=2, default='pac_predset')
    # parser.add_argument('--train_predset.rerun', action='store_true')
    # parser.add_argument('--train_predset.load_final', action='store_true')
    # parser.add_argument('--train_predset.binary_search', action='store_true')
    # parser.add_argument('--train_predset.bnd_type', type=str, default='direct')

    # parser.add_argument('--train_predset.T_step', type=float, default=1e-6) 
    # parser.add_argument('--train_predset.T_end', type=float, default=np.inf)
    # parser.add_argument('--train_predset.eps_tol', type=float, default=1.5)
        
    # parser.add_argument('--train_predset.n_bins', type=int, default=2)
    # parser.add_argument('--train_predset.bins', type=float, nargs='*', default=None)#default=[0.0, 0.8, np.inf])
    # parser.add_argument('--train_predset.n_iters', type=int, default=100)

    # parser.add_argument('--train_predset.B', type=int, default=1000)

    
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    #args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')

    ## dataset specific parameters
    if 'Normal' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 2
        if args.data.dim is None:
            args.data.dim = [2048]
        if args.model.base is None:
            args.model.base = 'Linear'
        if args.model.feat_dim is None:
            assert(len(args.data.dim) == 1)
            args.model.feat_dim = args.data.dim[0]
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True
        if args.data.n_train_src is None:
            args.data.n_train_src = 50000
        if args.data.n_train_tar is None:
            args.data.n_train_tar = args.data.n_train_src
        if args.data.n_val_src is None:
            args.data.n_val_src = 50000
        if args.data.n_val_tar is None:
            args.data.n_val_tar = args.data.n_val_src
        if args.data.n_test_src is None:
            args.data.n_test_src = 50000
        if args.data.n_test_tar is None:
            args.data.n_test_tar = args.data.n_test_src

        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_val_src
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
        # if args.model_iw_max.n is None:
        #     args.model_iw_max.n = args.data.n_val_tar
        # else:
        #     assert(args.model_iw_max.n <= args.data.n_val_tar)

    elif 'MNIST' in args.data.src:
        raise NotImplementedError
        if args.data.n_labels is None:
            args.data.n_labels = 10
        if args.data.img_size is None:
            args.data.img_size = (3, 32, 32)
        if args.model.base is None:
            args.model.base = 'ResNet18'
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'
        if args.model.sd is None:
            args.model.sd = 'MidFNN'
        if args.model.feat_dim is None:
            args.model.feat_dim = 512
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True
        if args.data.n_train_src is None:
            args.data.n_train_src = 50000
        if args.data.n_train_tar is None:
            args.data.n_train_tar = args.data.n_train_src
        if args.data.n_val_src is None:
            args.data.n_val_src = 10000
        # if args.data.n_val_tar is None:
        #     args.data.n_val_tar = args.data.n_val_src
        # if args.data.n_test_src is None:
        #     args.data.n_test_src = 10000
        # if args.data.n_test_tar is None:
        #     args.data.n_test_tar = args.data.n_test_src

        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_val_src
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
        # if args.model_iw_max.n is None:
        #     args.model_iw_max.n = args.data.n_val_tar
        # else:
        #     assert(args.model_iw_max.n <= args.data.n_val_tar)
            
    elif 'DomainNet' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 345
        if args.data.img_size is None:
            args.data.img_size = (3, 224, 224)
        if args.model.base is None:
            args.model.base = 'ResNet101'
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True
        ## data
        if args.data.n_val_src is None:
            args.data.n_val_src = 20000
        # if args.data.n_val_tar is None:
        #     args.data.n_val_tar = args.data.n_val_src
        ## models
        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_val_src
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
        # if args.model_iw_max.n is None:
        #     args.model_iw_max.n = args.data.n_val_tar
        # else:
        #     assert(args.model_iw_max.n <= args.data.n_val_tar)

            
    elif 'ImageNet' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 1000
        if args.data.img_size is None:
            args.data.img_size = (3, 224, 224)
        if args.model.base is None:
            args.model.base = 'ResNet101'
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
        if args.model.path_pretrained is None:
            args.model.path_pretrained = 'pytorch'
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True

        if args.data.n_val_src is None:
            args.data.n_val_src = 25000
        if args.data.n_val_tar is None:
            args.data.n_val_tar = args.data.n_val_src
        # if args.data.n_test_src is None:
        #     args.data.n_test_src = 20000 # use relatively small test set for speed up evaluation
        # if args.data.n_test_tar is None:
        #     args.data.n_test_tar = args.data.n_test_src

            
        if args.model_predset.n is None:
            args.model_predset.n = args.data.n_val_src
        else:
            assert(args.model_predset.n <= args.data.n_val_src)
        # if args.model_iw_max.n is None:
        #     args.model_iw_max.n = args.data.n_val_tar
        # else:
        #     assert(args.model_iw_max.n <= args.data.n_val_tar)

    else:
        raise NotImplementedError    
        
    # if args.train_predset.bins is not None:
    #     assert(len(args.train_predset.bins) == args.train_predset.n_bins+1)
    # print("bins =", args.train_predset.bins)

    
    ## print args
    util.print_args(args)
    
    ## setup logger
    #os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    #sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    plot(args)


