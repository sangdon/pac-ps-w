import sys, os
import pickle
import glob

import torch as tc
#from torchvision import transforms as tforms
#from torchvision.datasets.folder import default_loader

import model
#import data.custom_transforms as ctforms
from imagenet import ImageNet, ImageNetC


if __name__ == '__main__':
    ## parameters
    model_name = 'ResNet101'
    n_labels = 1000    
    feat_name = 'feat_da_resnet101'
    root_src = 'imagenet'
    root_tar = 'imagenetc'
    model_path = '/home/sangdonp/models/ImageNet/imagenet_src_ImageNet_tar_ImageNetC_dann/model_params_final_no_adv'
    batch_size = 800
    
    ## init loaders
    dsld_src = ImageNet(root_src, batch_size=batch_size,
                        train_rnd=False, val_rnd=False, test_rnd=False, return_path=True)
    dsld_tar = ImageNetC(root_tar, batch_size=batch_size,
                        train_rnd=False, val_rnd=False, test_rnd=False, return_path=True)
    
    ## load a model
    mdl = getattr(model, model_name)(n_labels=n_labels, path_pretrained=model_path)
    mdl = tc.nn.DataParallel(mdl)
    mdl.cuda()
    mdl.eval()

    ## get features
    for root, dsld in zip([root_src, root_tar], [dsld_src, dsld_tar]):
        for split_name in ['test', 'val', 'train']:
            for fn, x, _ in getattr(dsld, split_name):

                i_missing = []
                fn_new_list = []
                for fn_i in fn:
                    fn_new = os.path.join(root + '_' + feat_name, '/'.join(fn_i.split('/')[1:])) + '.feat'
                    if not os.path.exists(fn_new):
                        i_missing.append(True)
                        fn_new_list.append(fn_new)
                    else:
                        i_missing.append(False)
                if any(i_missing):
                    x = x[tc.tensor(i_missing)]
                else:
                    continue
                
                with tc.no_grad():
                    feat = mdl(x.cuda())['feat'].squeeze().cpu().detach().numpy()

                ## save the feature
                for fn_new, feat_i in zip(fn_new_list, feat):
                    #fn_new = os.path.join(root + '_' + feat_name, '/'.join(fn_i.split('/')[1:])) + '.feat'
                    os.makedirs(os.path.dirname(fn_new), exist_ok=True)
                    pickle.dump(feat_i, open(fn_new, 'wb'))

                    print(fn_new, feat_i.shape, feat_i.mean())


    
    
