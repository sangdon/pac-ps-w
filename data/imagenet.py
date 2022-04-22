import os, sys
import glob
import time
import random
import pickle

import torch as tc
from torchvision import transforms as tforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader

from torchvision.datasets.utils import check_integrity
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import data
import data.custom_transforms as ctforms


def _load_meta_file(meta_file):
    if check_integrity(meta_file):
        return tc.load(meta_file)
    else:
        raise RuntimeError("Meta file not found or corrupted.")

    
def label_to_name(root, label_to_wnid):
    meta_file = os.path.join(root, 'meta.bin')
    wnid_to_names = _load_meta_file(meta_file)[0]

    names = [wnid_to_names[wnid][0].replace(' ', '_').replace('\'', '_') for wnid in label_to_wnid]
    return names


class ImageNet(data.ClassificationData):
    def __init__(
            self, root, batch_size,
            #image_size=IMAGE_SIZE, color=False,
            train_rnd=True, val_rnd=False, test_rnd=False,
            train_aug=False, val_aug=False, test_aug=False,
            aug_types=[],
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            num_workers=4,
            normalize=True,
            return_path=False,
            load_feat=None,
            **kwargs,
    ):
        if load_feat is not None:
            root = root + '_feat_' + load_feat
            
        ## default tforms
        if load_feat is not None:
            tforms_dft = []
            tforms_dft_rnd = []
        else:
            tforms_dft = [
                ctforms.Resize(256),
                ctforms.CenterCrop(224),
                ctforms.ToTensor(),
                ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctforms.Identity(),
            ]
            tforms_dft_rnd = [
                ctforms.RandomResizedCrop(224) if load_feat is None else ctforms.Identity(),
                ctforms.RandomHorizontalFlip() if load_feat is None else ctforms.Identity(),
                ctforms.ToTensor(),
                ctforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctforms.Identity(),
            ]

        loader = default_loader if load_feat is None else lambda fn: pickle.load(open(fn, 'rb'))

        super().__init__(
            root=root, batch_size=batch_size,
            dataset_fn=data.ImageList,
            sample_size=sample_size,
            train_rnd=train_rnd, val_rnd=val_rnd, test_rnd=test_rnd,
            train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
            aug_types=aug_types,
            num_workers=num_workers,
            tforms_dft=tforms_dft, tforms_dft_rnd=tforms_dft,
            ext='JPEG' if load_feat is None else 'feat',
            seed=seed,
            return_path=return_path,
            loader=loader,
        )

        ## add id-name map
        self.names = label_to_name(root, self.test.dataset.dataset.classes)

        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        

class ImageNetCDataset(DatasetFolder):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None,
                 n_data=None,
                 seed=None,
                 return_path=False,
                 extensions=('JPEG',),
    ):
        super().__init__(root=root, transform=transform, target_transform=target_transform, loader=loader, extensions=extensions,
                         is_valid_file=is_valid_file)
        self.n_data = len(self.samples) if n_data is None else n_data
        self.seed = seed
        self.return_path = return_path

        ## shuffle and truncate data
        random.seed(seed)
        i_rnd = [i for i in range(len(self.samples))]
        random.shuffle(i_rnd)
        self.samples = [self.samples[i] for i in i_rnd[:self.n_data]]
        self.targets = [self.targets[i] for i in i_rnd[:self.n_data]]
        #self.imgs = self.samples
        random.seed(int(time.time()))

        
    def _find_classes(self, dir):
        dir_new = os.path.join(dir, 'gaussian_noise', '1')
        assert(os.path.exists(dir_new))
        classes = [d.name for d in os.scandir(dir_new) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    
    @staticmethod
    def make_dataset(
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None):

        assert(len(extensions) == 1)
        fns = glob.glob(os.path.join(directory, '**', '**', '**', f'*.{extensions[0]}'))

        instances = [(fn, class_to_idx[fn.split('/')[-2]]) for fn in fns]
        return instances


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return path, sample, target
        else:            
            return sample, target
        

class ImageNetC:
    def __init__(
            self, root, batch_size,
            #image_size=None, color=True,
            train_rnd=True, val_rnd=False, test_rnd=False, ## train is randomized; it conducts basic random augmentations
            #train_aug=False, val_aug=False, test_aug=False,
            #normalize=True,
            #aug_types=[],
            num_workers=4,
            #domain_label=None,
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            return_path=False,
            load_feat=None,
            normalize=True,
            **kwargs,
    ):
        if load_feat is not None:
            root = root + '_feat_' + load_feat

        ## default tforms
        if load_feat is not None:
            tforms_dft = None
        else:
            tforms_dft = [
                #tforms.Resize(256), # already applied
                #tforms.CenterCrop(224), # already applied
                tforms.ToTensor(),
                tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else ctforms.Identity(),
            ]
            tforms_dft = tforms.Compose(tforms_dft)

        ## data loaders
        loader = default_loader if load_feat is None else lambda fn: pickle.load(open(fn, 'rb'))
        extensions = ('JPEG',) if load_feat is None else ('feat',)
        self.train = tc.utils.data.DataLoader(
            ImageNetCDataset(os.path.join(root, 'train'), tforms_dft, n_data=sample_size['train'],
                             seed=seed, return_path=return_path, loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers, pin_memory=True)
        self.val = tc.utils.data.DataLoader(
            ImageNetCDataset(os.path.join(root, 'val'), tforms_dft, n_data=sample_size['val'],
                             seed=seed, return_path=return_path, loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers, pin_memory=True)
        self.test = tc.utils.data.DataLoader(
            ImageNetCDataset(os.path.join(root, 'test'), tforms_dft, n_data=sample_size['test'],
                             seed=seed, return_path=return_path, loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers, pin_memory=True)

        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        ## add id-name map
        self.names = label_to_name(root, self.test.dataset.classes)

        
class ImageNetAdvExDataset(DatasetFolder):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 loader=lambda path: pickle.load(open(path, 'rb')),
                 is_valid_file=None,
                 n_data=None,
                 seed=None,
                 return_path=False,
                 extensions=('.advex',)
    ):
        super().__init__(root=root, extensions=extensions,
                         transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        self.n_data = len(self.samples) if n_data is None else n_data
        self.seed = seed
        self.return_path = return_path

        ## shuffle and truncate data
        random.seed(seed)
        i_rnd = [i for i in range(len(self.samples))]
        random.shuffle(i_rnd)
        self.samples = [self.samples[i] for i in i_rnd[:self.n_data]]
        self.targets = [self.targets[i] for i in i_rnd[:self.n_data]]
        self.imgs = self.samples
        random.seed(int(time.time()))

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return path, sample, target
        else:            
            return sample, target


class ImageNetAdvEx:
    def __init__(
            self, root, batch_size,
            #image_size=None, color=True,
            train_rnd=True, val_rnd=False, test_rnd=False, ## train is randomized; it conducts basic random augmentations
            #train_aug=False, val_aug=False, test_aug=False,
            #normalize=True,
            #aug_types=[],
            num_workers=4,
            #domain_label=None,
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            return_path=False,
            load_feat=None,
            **kwargs,
    ):
        ## default tforms
        class CustomToTensor:
            def __call__(self, x):
                x = tc.tensor(x, dtype=tc.float)
                x = x / 255.0
                return x
        tforms_dft = [
            #tforms.Resize(256), # already applied
            #tforms.CenterCrop(224), # already applied
            CustomToTensor(),
            #tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # will be applied later
        ]

        if load_feat is None:            
            tforms_dft = tforms.Compose(tforms_dft)
            loader = lambda fn: pickle.load(open(fn, 'rb'))
            extensions = ('advex',) 
        else:
            root = root + '_feat_' + load_feat
            tforms_dft = None
            loader = lambda fn: pickle.load(open(fn, 'rb'))
            extensions = ('feat',)
            
        ## data loaders
        self.train = tc.utils.data.DataLoader(
            ImageNetAdvExDataset(os.path.join(root, 'train'), tforms_dft, n_data=sample_size['train'], seed=seed, return_path=return_path,
                                 loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers, pin_memory=True)
        self.val = tc.utils.data.DataLoader(
            ImageNetAdvExDataset(os.path.join(root, 'val'), tforms_dft, n_data=sample_size['val'], seed=seed, return_path=return_path,
                                 loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers, pin_memory=True)
        self.test = tc.utils.data.DataLoader(
            ImageNetAdvExDataset(os.path.join(root, 'test'), tforms_dft, n_data=sample_size['test'], seed=seed, return_path=return_path,
                                 loader=loader, extensions=extensions),
            batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers, pin_memory=True)

        print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        ## add id-name map
        self.names = label_to_name(root, self.test.dataset.classes)


ImageNetAdvEx_eps_0p01 = ImageNetAdvEx
        
if __name__ == '__main__':
    #dsld = data.ImageNet('data/imagenet', 100)
    #dsld = data.ImageNetAdvEx('data/imagenet_advex_resnet101', 100, sample_size={'train': 1000, 'val': 2000, 'test': 3000})
    dsld = data.ImageNetC(root='data/imagenetc', batch_size=100, sample_size={'train': None, 'val': None, 'test': None})
    
    # import torchvision 
    # for x, y in dsld.val:
    #     print(x.shape)
    #     [print(x_i.min(), x_i.max(), x_i.mean()) for x_i in x]
    #     torchvision.utils.save_image(x, 'img1.png')
    #     break
    
    # print("#train = ", data.compute_num_exs(dsld.train, verbose=True))
    # print("#val = ", data.compute_num_exs(dsld.val))
    # print("#test = ", data.compute_num_exs(dsld.test))

## ImageNet
#train =  1281167
#val =  25000
#test =  25000



