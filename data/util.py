import os, sys
import numpy as np
import time
import glob
import random
import math
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
import pickle
import warnings

import torch as tc
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset, Subset

from data import get_aug_tforms


"""
Simple wrapper functions
"""
def compute_num_exs(ld, verbose=False):
    n = 0
    t = time.time()
    for x, _ in ld:
        n += x.shape[0]
        if verbose:
            print("[%f sec.] n = %d"%(time.time()-t, n))
            t = time.time()
    return n


"""
functions/classes for data loaders
"""
    
def shuffle_list(list_ori, seed):
    random.seed(seed)
    random.shuffle(list_ori)
    random.seed(int(time.time()))
    return list_ori


def get_random_index(n_ori, n, seed):
    random.seed(seed)
    if n_ori < n:
        index = [random.randint(0, n_ori-1) for _ in range(n)]
    else:
        index = [i for i in range(n_ori)]
        random.shuffle(index)
        index = index[:n]
        
    random.seed(time.time())
    return index


def find_classes(root):
    classes = [d.name for s in ['train', 'val', 'test'] for d in os.scandir(os.path.join(root, s)) if d.is_dir()]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_class_name(fn):
    return fn.split('/')[-2]


def make_dataset(fn_list, class_to_idx):
    instances = []
    for fn in fn_list:
        class_idx = class_to_idx[get_class_name(fn)]
        item = fn, class_idx
        instances.append(item)
    return instances

    
class ImageList:
    def __init__(self, fn_list, classes, class_to_idx, transform=None, loader=default_loader, return_path=False):
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = make_dataset(fn_list, class_to_idx)
        self.return_path = return_path

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = target 
        sample = self.loader(path)
        if self.transform is not None:
            sample, target = self.transform((sample, target))
        if self.return_path:
            return path, sample, target
        else:
            return sample, target


    def __len__(self):
        return len(self.samples)


def get_tforms(tforms_dft, tforms_dft_rnd, train_rnd, val_rnd, test_rnd, train_aug, val_aug, test_aug, aug_types):
    ## get data augmentation tforms
    tforms_aug = get_aug_tforms(aug_types)

    ## set tforms for each data split
    tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
    tforms_train = tforms_train + tforms_aug if train_aug else tforms_train
    tforms_val = tforms_dft_rnd if val_rnd else tforms_dft
    tforms_val = tforms_val + tforms_aug if val_aug else tforms_val
    tforms_test = tforms_dft_rnd if test_rnd else tforms_dft
    tforms_test = tforms_test + tforms_aug if test_aug else tforms_test

    return tforms_train, tforms_val, tforms_test
    

class ClassificationData:
    def __init__(self, root, batch_size,
                 dataset_fn,
                 sample_size,
                 train_rnd, val_rnd, test_rnd,
                 train_aug, val_aug, test_aug,
                 aug_types,
                 num_workers,
                 tforms_dft, tforms_dft_rnd,
                 ext,
                 seed,
                 return_path,
                 loader=default_loader,
    ):
        ## get data augmentation tforms
        tforms_aug = get_aug_tforms(aug_types)

        ## set tforms for each data split
        tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
        tforms_train = tforms_train + tforms_aug if train_aug else tforms_train
        tforms_val = tforms_dft_rnd if val_rnd else tforms_dft
        tforms_val = tforms_val + tforms_aug if val_aug else tforms_val
        tforms_test = tforms_dft_rnd if test_rnd else tforms_dft
        tforms_test = tforms_test + tforms_aug if test_aug else tforms_test

        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)

        ## get class name
        classes, class_to_idx = find_classes(root)

        ## splits
        split_list = split_data_cls(sample_size, root, ext, seed)
        
        ## create loaders
        ds = Subset(dataset_fn(split_list['train'], classes, class_to_idx, transform=transforms.Compose(tforms_train), return_path=return_path, loader=loader),
                    get_random_index(len(split_list['train']),
                                     len(split_list['train']) if sample_size['train'] is None else sample_size['train'],
                                     seed))
        self.train = DataLoader(ds, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers)
        ds = Subset(dataset_fn(split_list['val'], classes, class_to_idx, transform=transforms.Compose(tforms_train), return_path=return_path, loader=loader),
                    get_random_index(len(split_list['val']), len(split_list['val']) if sample_size['val'] is None else sample_size['val'], seed))
        self.val = DataLoader(ds, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers)
        ds = Subset(dataset_fn(split_list['test'], classes, class_to_idx, transform=transforms.Compose(tforms_train), return_path=return_path, loader=loader),
                    get_random_index(len(split_list['test']), len(split_list['test']) if sample_size['test'] is None else sample_size['test'], seed))
        self.test = DataLoader(ds, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers)
        



"""
domain data loader
"""
class JointLoader:
    def __init__(self, lds, domain_labels=None, both_labels=False, truncate=False):
        self.lds = lds
        self.domain_labels = domain_labels if domain_labels is not None else [None]*len(lds)
        self.both_labels = both_labels
        self.truncate = truncate

        
    def __iter__(self):
        self.iters = [iter(ld) for ld in self.lds]
        self.iter_end = [False for ld in self.lds]
        return self
    
    def __next__(self):
        x_list, y_list, y_dom_list = [], [], []
        for i, (it, label) in enumerate(zip(self.iters, self.domain_labels)):
            if self.truncate:
                x, y = next(it)
            else:
                try:
                    x, y = next(it)
                except StopIteration:
                    self.iter_end[i] = True

                    if all(self.iter_end):
                        raise StopIteration
                    else:
                        self.iters[i] = iter(self.lds[i])
                        x, y = next(self.iters[i])
            x_list.append(x)
            y_list.append(y)
            
            if label is not None:
                y_dom = tc.ones_like(y)*label
                y_dom_list.append(y_dom)
        
        # maintain the same batch size
        bs_min = min([o.shape[0] for o in x_list])
        x_list = [o[:bs_min] for o in x_list]
        x_list = tc.cat(x_list, 0)
        y_list = [o[:bs_min] for o in y_list]
        y_list = tc.cat(y_list, 0)
        if y_dom_list:
            y_dom_list = [o[:bs_min] for o in y_dom_list]
            y_dom_list = tc.cat(y_dom_list, 0)
            if self.both_labels:
                return x_list, (y_list, y_dom_list)
            else:
                return x_list, y_dom_list
        else:
            return x_list, y_list

    
class DomainData:
    def __init__(self, dsld_src, dsld_tar, truncate=False):
        self.train = JointLoader([dsld_src.train, dsld_tar.train], [1, 0], truncate=truncate)
        self.val = JointLoader([dsld_src.val, dsld_tar.val], [1, 0], truncate=truncate)
        self.test = JointLoader([dsld_src.test, dsld_tar.test], [1, 0], truncate=truncate)


class DAData:
    def __init__(self, dsld_src, dsld_tar, truncate=False):
        self.train = JointLoader([dsld_src.train, dsld_tar.train], [1, 0], both_labels=True, truncate=truncate)
        self.val = JointLoader([dsld_src.val, dsld_tar.val], [1, 0], both_labels=True, truncate=truncate)
        self.test = JointLoader([dsld_src.test, dsld_tar.test], [1, 0], both_labels=True, truncate=truncate)
        
