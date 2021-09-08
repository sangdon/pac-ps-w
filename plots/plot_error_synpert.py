import os, sys

import torch as tc
#from torchvision import transforms as tforms
#from torchvision.datasets.folder import default_loader

import model
#import data.custom_transforms as ctforms
from data import ImageNet, ImageNetC


if __name__ == '__main__':
    ## parameters
    model_name = 'ResNet101'
    n_labels = 1000    
    root_src = 'data/imagenet'
    root_tar = 'data/imagenetc'
    model_path = '/home/sangdonp/models/ImageNet/imagenet_src_ImageNet_tar_ImageNetC_dann_35/model_params_final_no_adv'    
    batch_size = 300
    
    ## init loaders
    dsld_src = ImageNet(root_src, batch_size=batch_size,
                        train_rnd=False, val_rnd=False, test_rnd=False, return_path=True)
    dsld_tar = ImageNetC(root_tar, batch_size=batch_size,
                        train_rnd=False, val_rnd=False, test_rnd=False, return_path=True)
    
    ## load a model
    mdl_ori = getattr(model, model_name)(n_labels=n_labels, path_pretrained='pytorch')
    mdl_ori = tc.nn.DataParallel(mdl_ori)
    mdl_ori.cuda()
    mdl_ori.eval()

    mdl = getattr(model, model_name)(n_labels=n_labels, path_pretrained=model_path)
    mdl = tc.nn.DataParallel(mdl)
    mdl.cuda()
    mdl.eval()

    ## check error
    for mdl_name, mdl_i in zip(['naive', 'da'], [mdl_ori, mdl]):
        for domain_name, ld in zip(['src', 'tar'], [dsld_src.test, dsld_tar.test]):
            n, n_error = 0.0, 0.0
            for _, x, y in ld:
                x, y = x.cuda(), y.cuda()
                with tc.no_grad():
                    yh = mdl_i(x)['yh_top']
                n_error += (y!=yh).sum()
                n += x.shape[0]
            print(f'{domain_name} error = {int(n_error)} / {int(n)} = {n_error / n * 100.0}%')

