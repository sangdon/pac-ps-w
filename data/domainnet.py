import os, sys
import numpy as np
import warnings
import pickle

import torch as tc
from torchvision import transforms as tforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import default_loader

import data
import data.custom_transforms as ctforms

IMAGE_SIZE=224
DOMAIN_LIST=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

class DomainNetDataset:
    def __init__(self, fn_label_list, transform, loader=default_loader, return_path=False):
        self.fn_label_list = fn_label_list
        self.transform = transform
        self.loader = loader
        self.return_path = return_path


    def __getitem__(self, index):
        path, target = self.fn_label_list[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample, target = self.transform((sample, target))
        if self.return_path:
            return path, sample, target
        else:
            return sample, target


    def __len__(self):
        return len(self.fn_label_list)

    
class DomainNet:
    def __init__(
            self, root=None, batch_size=64,
            image_size=IMAGE_SIZE, normalize=True,
            train_rnd=True, val_rnd=True, test_rnd=False,
            train_aug=False, val_aug=False, test_aug=False,
            aug_types=[],
            color=True,
            sample_size={'train': None, 'val': 5000, 'test': None},
            seed=0,
            num_workers=4,
            load_feat=None,
            return_path=False,
            **kwargs,
    ):
        ## tforms
        if load_feat:
            tforms_train = None
            tforms_val = None
            tforms_test = None
        else:

            # use imagenet default tforms
            tforms_dft = [
                ctforms.Resize(256),
                ctforms.CenterCrop(image_size),
                ctforms.ToTensor(),
                ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctform.Identity(),
            ]
            tforms_dft_rnd = [
                ctforms.RandomResizedCrop(image_size),
                ctforms.RandomHorizontalFlip(),
                ctforms.ToTensor(),
                ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctform.Identity(),
            ]

            tforms_train, tforms_val, tforms_test = data.get_tforms(
                tforms_dft, tforms_dft_rnd,
                train_rnd, val_rnd, test_rnd,
                train_aug, val_aug, test_aug, aug_types)
            tforms_train = tforms.Compose(tforms_train)
            tforms_val = tforms.Compose(tforms_test)
            tforms_test = tforms.Compose(tforms_test)

        ## load data
        train_fn_label = [(l.split(' ')[0], int(l.split(' ')[1].strip()))
                          for r in root for l in open(r+'_train%s.txt'%(f'_feat_{load_feat}' if load_feat else ''), 'r').readlines()]
        val_fn_label = [(l.split(' ')[0], int(l.split(' ')[1].strip()))
                        for r in root for l in open(r+'_val_split%s.txt'%(f'_feat_{load_feat}' if load_feat else ''), 'r').readlines()]
        test_fn_label = [(l.split(' ')[0], int(l.split(' ')[1].strip()))
                         for r in root for l in open(r+'_test_split%s.txt'%(f'_feat_{load_feat}' if load_feat else ''), 'r').readlines()]
        train_fn_label = data.shuffle_list(train_fn_label, seed)
        val_fn_label = data.shuffle_list(val_fn_label, seed)
        test_fn_label = data.shuffle_list(test_fn_label, seed)

        ## subsample data
        if sample_size['train'] is not None:
            assert(len(train_fn_label) >= sample_size['train'])
            train_fn_label = train_fn_label[:sample_size['train']]
        if sample_size['val'] is not None:
            assert(len(val_fn_label) >= sample_size['val'])
            val_fn_label = val_fn_label[:sample_size['val']]
        if sample_size['test'] is not None:
            assert(len(test_fn_label) >= sample_size['test'])
            test_fn_label = test_fn_label[:sample_size['test']]
            
        ## add root on file names
        assert(all([os.path.dirname(root[0]) == p for p in [os.path.dirname(r) for r in root]]))
        train_fn_label = [(os.path.join(os.path.dirname(root[0]), fn), l) for fn, l in train_fn_label]
        val_fn_label = [(os.path.join(os.path.dirname(root[0]), fn), l) for fn, l in val_fn_label]
        test_fn_label = [(os.path.join(os.path.dirname(root[0]), fn), l) for fn, l in test_fn_label]

        loader = (lambda x: pickle.load(open(x, 'rb'))) if load_feat else default_loader
        
        ## init data loader
        ds = DomainNetDataset(train_fn_label, transform=tforms_train, loader=loader, return_path=return_path)
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers)
        ds = DomainNetDataset(val_fn_label, transform=tforms_val, loader=loader, return_path=return_path)
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers)
        ds = DomainNetDataset(test_fn_label, transform=tforms_test, loader=loader, return_path=return_path)
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers)

        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        ## load the meta file
        meta = pickle.load(open(os.path.join(os.path.dirname(root[0]), 'meta'), 'rb'))
        self.id_name_map = meta['id_name_map']
        self.name_id_map = meta['name_id_map']

        
class DomainNetClipart(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('clipart', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'clipart')]
        
        super().__init__(*args, **kwargs)

        
class DomainNetInfograph(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('infograph', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'infograph')]
        
        super().__init__(*args, **kwargs)

        
class DomainNetPainting(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('painting', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'painting')]

        super().__init__(*args, **kwargs)


class DomainNetQuickdraw(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('quickdraw', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'quickdraw')]

        super().__init__(*args, **kwargs)

        
class DomainNetReal(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('real', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'real')]

        super().__init__(*args, **kwargs)


class DomainNetSketch(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('sketch', '')
        kwargs['root'] = [os.path.join(kwargs['root'], 'sketch')]

        super().__init__(*args, **kwargs)


class DomainNetAll(DomainNet):
    def __init__(self, *args, **kwargs):
        kwargs['root'] = kwargs['root'].replace('all', '')
        kwargs['root'] = [os.path.join(kwargs['root'], dn) for dn in DOMAIN_LIST]

        super().__init__(*args, **kwargs)
        
        
if __name__ == '__main__':
    dsld = data.DomainNetAll(root='data/domainnet', batch_size=100, sample_size={'train': None, 'val': None, 'test': None})
    print("#train =", data.compute_num_exs(dsld.train))
    print("#val =", data.compute_num_exs(dsld.val))
    print("#test =", data.compute_num_exs(dsld.test))

"""
"""
