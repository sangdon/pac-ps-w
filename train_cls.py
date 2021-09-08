import os, sys
import argparse
import warnings
import math

import torch as tc

import util
import data
import model
import learning
import uncertainty

def main(args):

    ## init datasets
    print("## init source datasets: %s"%(args.data.src))
    ds_src = getattr(data, args.data.src)(
        root=os.path.join('data', args.data.src.lower()),
        batch_size=args.data.batch_size,
        dim=args.data.dim,
        train_rnd=True, val_rnd=False, test_rnd=False, 
        train_aug=args.data.aug_src is not None, val_aug=args.data.aug_src is not None, test_aug=args.data.aug_src is not None,
        aug_types=args.data.aug_src,
        color=True if args.data.dim[0]==3 else False,
        num_workers=args.data.n_workers,
        sample_size={'train': args.data.n_train_src, 'val': args.data.n_val_src, 'test': args.data.n_test_src},
        seed=args.data.seed,
        normalize=not args.model.normalize,
        load_feat=args.data.load_feat,
    )
    print()
    
    print("## init target datasets: %s"%(args.data.tar))
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        dim=args.data.dim,
        train_rnd=True, val_rnd=False, test_rnd=False,
        train_aug=args.data.aug_tar is not None, val_aug=args.data.aug_tar is not None, test_aug=args.data.aug_tar is not None,
        aug_types=args.data.aug_tar,
        color=True if args.data.dim[0]==3 else False,
        num_workers=args.data.n_workers,
        sample_size={'train': args.data.n_train_tar, 'val': args.data.n_val_tar, 'test': args.data.n_test_tar},
        seed=args.data.seed,
        normalize=not args.model.normalize,
        load_feat=args.data.load_feat,
    )
    print()

    if args.train.method == 'DANN':
        print("## init domain adaptation dataset: src = %s, tar = %s"%(args.data.src, args.data.tar))
        ds_da = data.DAData(ds_src, ds_tar, truncate=args.data.truncate_da)
        print()

    print("## init domain datasets: src = %s, tar = %s"%(args.data.src, args.data.tar))
    ds_dom = data.DomainData(ds_src, ds_tar, truncate=args.data.truncate_da)
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=args.data.dim[0], n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    else:
        raise NotImplementedError
    
    if args.data.load_feat:
        print("## init models: %s"%(args.model.base_feat))
        mdl = getattr(model, args.model.base_feat)(mdl)
    print()

    if args.model.normalize:
        print('## init an image normalizer as a pre-processing model')
        mdl = model.ExampleNormalizer(mdl)

    if args.train.method == 'DANN':
        print("## init models for adv: %s"%(args.model.adv))
        mdl_adv = getattr(model, args.model.adv)(args.model.feat_dim, 1)
        print("## init models for DANN")
        mdl = model.DANN(mdl, mdl_adv)

    if args.multi_gpus:
        mdl = tc.nn.DataParallel(mdl).cuda()
    print()

    ## learning
    if args.train.method == 'src':
        l = learning.ClsLearner(mdl, args.train)
        if not args.model.pretrained:
            print("## train over source...")
            l.train(ds_src.train, ds_src.val)
    elif args.train.method == 'DANN':
        l = learning.ClsDALearner(mdl, args.train)
        print(f"## train using {args.train.method}...")
        l.train(ds_da.train, ld_test=ds_tar.test) # no model selection
    elif args.train.method == 'skip':
        l = learning.ClsLearner(mdl, args.train)
    else:
        raise NotImplementedError
    print("## test...")
    l.test(ds_src.test, ld_name=f'{args.data.src} (src)', verbose=True)
    l.test(ds_tar.test, ld_name=f'{args.data.tar} (tar)', verbose=True)
    print()
        

    ## iw learning
    if args.train_iw:
        
        ## init a model
        print("## init models for iw: %s"%(args.model.sd))
        mdl_sd = model.SourceDisc(getattr(model, args.model.sd)(args.model.feat_dim, 2), mdl)
        print()

        ## learning
        l = learning.ClsLearner(mdl_sd, args.train_sd, name_postfix='srcdisc')
        print("## train...")
        l.train(ds_dom.train, ds_dom.val)
        print("## test...")
        l.test(ds_dom.test, ld_name='domain dataset', verbose=True)
        print()

        ## init an IW model
        mdl_cal = model.NoCal(mdl_sd, cal_target=args.cal_sd.cal_target)
        mdl_iw = model.IW(mdl_cal, bound_type='mean') ## choose the uncalibrated iw
        mdl_iw.eval()

        ## estimate the maximum importance weight
        def estimate_iw_max(mdl_iw, ld, device):
            iw_list = []
            for x, y in ld:
                x = x.to(device)
                with tc.no_grad():
                    w = mdl_iw(x, y)
                iw_list.append(w)
            iw_list = tc.cat(iw_list)

            iw_sorted = iw_list.sort()[0]
            iw_max = iw_sorted[math.ceil(len(iw_list)*(1.0 - 0.01))]
            return iw_max
        
        iw_max = estimate_iw_max(mdl_iw, ds_src.train, args.device)
        print("# iw_max = %f"%(iw_max))

        ## compute effective sample size
        m_eff = uncertainty.estimate_eff_sample_size(ds_src.val, mdl_iw, args.device)
        print(f'## effective sample size over val = {m_eff}')

        ## plot iw
        print('## plot iw')
        uncertainty.plot_iw_wrapper(ds_src.train, mdl_iw, device=args.device,
                                    fn=os.path.join(args.snapshot_root, args.exp_name, 'figs', 'plot_iw_over_src'))


    
def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--multi_gpus', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--train_iw', action='store_true')
    parser.add_argument('--estimate', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=200)
    parser.add_argument('--data.n_workers', type=int, default=4)
    parser.add_argument('--data.src', type=str, required=True)
    parser.add_argument('--data.tar', type=str, required=True)
    parser.add_argument('--data.n_labels', type=int)
    #parser.add_argument('--data.img_size', type=int, nargs=3) ##TODO: img_size and dim are redundent
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
    parser.add_argument('--data.truncate_da', action='store_true')
    parser.add_argument('--data.load_feat', type=str)

    ## model args
    parser.add_argument('--model.base', type=str)
    parser.add_argument('--model.base_feat', type=str)
    parser.add_argument('--model.path_pretrained', type=str)
    parser.add_argument('--model.feat_dim', type=int)
    parser.add_argument('--model.sd', type=str, default='MidFNN')
    parser.add_argument('--model.adv', type=str, default='MidAdvFNN')
    parser.add_argument('--model.normalize', action='store_true')

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.resume', type=str)
    parser.add_argument('--train.method', type=str, default='src')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int, default=100)
    parser.add_argument('--train.lr', type=float, default=0.01)
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float, default=0.0)
    parser.add_argument('--train.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train.val_period', type=int, default=1)
    
    ## train args for a source discriminator
    parser.add_argument('--train_sd.rerun', action='store_true')
    parser.add_argument('--train_sd.resume', type=str)
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

    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
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


    elif 'MNIST' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 10
        if args.data.dim is None:
            args.data.dim = (3, 32, 32)
        if args.model.base is None:
            args.model.base = 'ResNet18'
        if args.model.feat_dim is None:
            args.model.feat_dim = 512
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        if args.data.n_train_src is None:
            args.data.n_train_src = 50000
        if args.data.n_train_tar is None:
            args.data.n_train_tar = args.data.n_train_src
        if args.data.n_val_src is None:
            args.data.n_val_src = 10000
        if args.data.n_val_tar is None:
            args.data.n_val_tar = args.data.n_val_src
        if args.data.n_test_src is None:
            args.data.n_test_src = 10000
        if args.data.n_test_tar is None:
            args.data.n_test_tar = args.data.n_test_src

    elif 'DomainNet' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 345
        if args.data.dim is None:
            args.data.dim = (3, 224, 224)
        if args.model.base is None:
            args.model.base = 'ResNet101'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
            
        if args.model.path_pretrained is None:
            args.model.pretrained = False
        else:
            args.model.pretrained = True

        # if args.data.n_train_src is None:
        #     args.data.n_train_src = 50000
        # if args.data.n_train_tar is None:
        #     args.data.n_train_tar = args.data.n_train_src
        # if args.data.n_val_src is None:
        #     args.data.n_val_src = 10000 # use relatively small val set for speed up training
        # if args.data.n_val_tar is None:
        #     args.data.n_val_tar = args.data.n_val_src
        if args.data.n_test_src is None:
            args.data.n_test_src = 5000 # use relatively small test set for speed up training (for when we want to compute test error during training)
        if args.data.n_test_tar is None:
            args.data.n_test_tar = args.data.n_test_src
        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'

            
    elif 'ImageNet' in args.data.src:
        if args.data.n_labels is None:
            args.data.n_labels = 1000
        if args.data.dim is None:
            args.data.dim = (3, 224, 224)
        if args.model.base is None:
            args.model.base = 'ResNet101'
        if args.model.feat_dim is None:
            args.model.feat_dim = 2048
        if args.model.path_pretrained is None:
            args.model.path_pretrained = 'pytorch'
        if args.model.path_pretrained  == 'pytorch':
            args.model.pretrained = True
        else:
            args.model.pretrained = False

        
        # if args.data.n_val_src is None:
        #     args.data.n_val_src = 25000
        # if args.data.n_val_tar is None:
        #     args.data.n_val_tar = args.data.n_val_src
        if args.data.n_test_src is None:
            args.data.n_test_src = 5000 # use relatively small test set for speed up training (for when we want to compute test error during training)
        if args.data.n_test_tar is None:
            args.data.n_test_tar = args.data.n_test_src

        if args.model.base_feat is None:
            args.model.base_feat = 'ResNetFeat'


    else:
        raise NotImplementedError    
        
    ## print args
    util.print_args(args)
    
    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)


