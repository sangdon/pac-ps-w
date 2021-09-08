import sys, os
import pickle

import torch as tc
from torchvision import transforms as tforms
from torchvision.datasets.folder import default_loader

import model
import data.custom_transforms as ctforms

IMAGE_SIZE=224
DOMAIN_LIST=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

if __name__ == '__main__':
    ## parameters
    model_name = 'ResNet101'
    n_labels = 345
    #model_path = '../snapshots/domainnetall_24/model_params_best'
    #model_path = '../snapshots/domainnet_all2sketch/model_params_final_no_adv'

    # # real
    # exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetReal_dann_long'
    # feat_name = f'feat_All2Real'

    # # clipart
    # exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetClipart_dann_long'
    # feat_name = f'feat_All2Clipart'

    # # painting
    # exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetPainting_dann_long'
    # feat_name = f'feat_All2Painting'

    # # quickdraw
    # exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetQuickdraw_dann_long'
    # feat_name = f'feat_All2Quickdraw'

    # # infograph
    # exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetInfograph_dann_long'
    # feat_name = f'feat_All2Infograph'

    # all
    exp_name = 'domainnet_src_DomainNetAll_tar_DomainNetAll_dann_long'
    feat_name = f'feat_All2All'

    model_path = os.path.expanduser(f'~/models/DomainNet/{exp_name}/model_params_final_no_adv')

    tforms_dft = tforms.Compose([
        ctforms.Resize(256),
        ctforms.CenterCrop(IMAGE_SIZE),
        ctforms.ToTensor(),
        ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    ## load a model
    mdl = getattr(model, model_name)(n_labels=n_labels)
    mdl.load_state_dict({k.replace('module.', '').replace('mdl.', ''): v for k, v in tc.load(model_path).items()})
    mdl.cuda()
    mdl.eval()

    ## get features
    for domain_name in DOMAIN_LIST:
        root = os.path.join('domainnet', domain_name)
        for split_name in ['train', 'val_split', 'test_split']:
            
            ## load file names
            fn_label = [(l.split(' ')[0], int(l.split(' ')[1].strip())) for l in open(root+f'_{split_name}.txt', 'r').readlines()]

            ## get a feature
            fn_label_new = []
            for fn, label in fn_label:
                fn = os.path.join('domainnet', fn)
                sample = default_loader(fn)
                sample, label = tforms_dft((sample, label))

                with tc.no_grad():
                    feat = mdl(sample.cuda().unsqueeze(0))['feat'].squeeze().cpu().detach().numpy()
                    
                ## save the feature
                root_new = root + '_' + feat_name
                fn_new_txt = fn.split('/')[-3]+'_'+feat_name + '/' + '/'.join(fn.split('/')[-2:]) + '.feat'
                fn_new = os.path.join(root_new, os.path.basename(os.path.dirname(fn)), os.path.basename(fn)+'.feat')
                os.makedirs(os.path.dirname(fn_new), exist_ok=True)
                pickle.dump(feat, open(fn_new, 'wb'))
                
                print(fn_new_txt, feat.shape, feat.mean())
                fn_label_new.append((fn_new_txt, str(label)+'\n'))
                
            ## save file names
            fn_label_new = [f"{t[0]} {t[1]}" for t in fn_label_new]
            open(root+f'_{split_name}_{feat_name}.txt', 'w').writelines(fn_label_new)
                


    
    
