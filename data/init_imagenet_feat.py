import sys, os
import pickle
import glob

import torch as tc
from torchvision import transforms as tforms
from torchvision.datasets.folder import default_loader

import model
import data.custom_transforms as ctforms

IMAGE_SIZE=224

if __name__ == '__main__':
    ## parameters
    model_name = 'ResNet101'
    n_labels = 1000    
    feat_name = f'feat_resnet101'
    root = 'imagenet'
    

    ## default transforms
    if root == 'imagenet':
        tforms_dft = tforms.Compose([
            ctforms.Resize(256),
            ctforms.CenterCrop(IMAGE_SIZE),
            ctforms.ToTensor(),
            ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'imagenetc' in root:
        tforms_dft = tforms.Compose([
            ctforms.ToTensor(),
            ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError
    
    ## load a model
    mdl = getattr(model, model_name)(n_labels=n_labels, path_pretrained='pytorch')
    mdl.cuda()
    mdl.eval()

    
    ## get features
    for split_name in ['val', 'test']: ## 'train'

        ## load file names
        fn_list = glob.glob(os.path.join(root, split_name, '**', '*.JPEG'), recursive=True)

        ## get a feature
        for fn in fn_list:
            sample = default_loader(fn)
            sample, _ = tforms_dft((sample, None))

            with tc.no_grad():
                feat = mdl(sample.cuda().unsqueeze(0))['feat'].squeeze().cpu().detach().numpy()
                
            ## save the feature
            fn_new = os.path.join(root + '_' + feat_name, '/'.join(fn.split('/')[1:])) + '.feat'
            os.makedirs(os.path.dirname(fn_new), exist_ok=True)
            pickle.dump(feat, open(fn_new, 'wb'))

            print(fn_new, feat.shape, feat.mean())


    
    
