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

def xywh2xyxy(xywh):
    xyxy = xywh.clone()
    if len(xyxy.size()) == 2:
        xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    else:
        xyxy[2:] = xywh[:2] + xywh[2:]
    return xyxy


def xyxy2xywh(xyxy):
    xywh = xyxy.clone()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh


"""
visualization functions
"""
def plot_bb(img, bb_xywh, fn=None):
    img_PIL = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img_PIL)
    draw.rectangle((*bb_xywh[:2], *(bb_xywh[:2]+bb_xywh[2:])), outline="white", width=2)
    if fn is not None:
        img_PIL.save(fn)
    else:
        return img_PIL


"""
functions/classes for data loaders
"""
    
def shuffle_list(list_ori, seed):
    random.seed(seed)
    random.shuffle(list_ori)
    random.seed(int(time.time()))
    return list_ori


def split_data_cls(sample_size, root, ext, seed):
    """
    randomly split val+test, but return train as it is
    """
    #assert(sample_size['train'] is None)
    
    fns_train = glob.glob(os.path.join(root, 'train', '**', '**.'+ext))
    fns_val = glob.glob(os.path.join(root, 'val', '**', '**.'+ext))
    fns_test = glob.glob(os.path.join(root, 'test', '**', '**.'+ext))
    
    ## shuffle
    fns_train = shuffle_list(fns_train, seed)
    fns_val = shuffle_list(fns_val, seed)
    fns_test = shuffle_list(fns_test, seed)

    # ## samples
    # n_val = len(fns_val) if sample_size['val'] is None else sample_size['val']
    # n_test = len(fns_test) if sample_size['test'] is None else sample_size['test']
    
    # ## truncate samples
    # fn_train_sampled = fns_train
    # fn_val_sampled = fns_val[:n_val]
    # fn_test_sampled = fns_test[:n_test]
    # assert(len(set(fn_val_sampled).intersection(set(fn_test_sampled))) == 0)

    # return {'train': fn_train_sampled, 'val': fn_val_sampled, 'test': fn_test_sampled}
    return {'train': fns_train, 'val': fns_val, 'test': fns_test}


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


# def init_loader_cls(dataset_fn, split_list, classes, class_to_idx, tforms, rnd, batch_size, num_workers):
#     dataset = dataset_fn(split_list, classes, class_to_idx, transform=transforms.Compose(tforms))
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=rnd, num_workers=num_workers)
#     return loader


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


class SubsetODD(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = []
        # odd_labels = np.array([self.dataset[i][1]%2==1 for i in range(len(self.dataset))])
        # self.indices = np.flatnonzero(odd_labels)
        warnings.warn('speed up')
        for i in range(len(self.dataset)):
            label = self.dataset[i][1]
            if label%2 == 1:
                self.indices.append(i)

                
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    
    def __len__(self):
        return len(self.indices)

    
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
        













        
#######################



# def default_loader(path: str) -> Any:
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)





# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        f.close()
    return img



## legacy?
def shuffle_initial_data_order(ld, seed):
    n_data = len(ld.dataset)
    np.random.seed(seed)
    idx_rnd = np.random.permutation(n_data)
    ##TODO: generalize
    if hasattr(ld.dataset, "samples"):
        ld.dataset.samples = [ld.dataset.samples[i] for i in idx_rnd]
    if hasattr(ld.dataset, "targets"):
        ld.dataset.targets = [ld.dataset.targets[i] for i in idx_rnd]
    if hasattr(ld.dataset, "imgs"):
        ld.dataset.imgs = [ld.dataset.imgs[i] for i in idx_rnd]
    if hasattr(ld.dataset, "frames_pair"):
        ld.dataset.frames_pair = [ld.dataset.frames_pair[i] for i in idx_rnd]
    if hasattr(ld.dataset, "fn"):
        ld.dataset.fn = [ld.dataset.fn[i] for i in idx_rnd]

    np.random.seed(int(time.time()%2**32))    




# def split_list(split_ratio, list_ori):
#     list_split = []
#     n_start = 0
#     for i, ratio in enumerate(split_ratio):
#         n = math.floor(len(list_ori)*ratio)
#         if i+1 == len(split_ratio):
#             list_split.append(list_ori[n_start:])
#         else:            
#             list_split.append(list_ori[n_start:n_start+n])
#         n_start += n
#     random.seed(time.time())
#     return list_split


    


def get_split_list(split_ratio, root, ext, seed): ##TODO: legacy function
    fns_train = glob.glob(os.path.join(root, 'train', '**', '**.'+ext)) ##TODO: 
    fns_val = glob.glob(os.path.join(root, 'val', '**', '**.'+ext))
    fns_test = glob.glob(os.path.join(root, 'test', '**', '**.'+ext))

    ## shuffle list since usually it's sorted
    random.seed(seed)
    random.shuffle(fns_train)
    random.seed(seed)
    random.shuffle(fns_val)
    random.seed(seed)
    random.shuffle(fns_test)

    ## set splits
    fns_split = []
    for name, ratio in split_ratio.items():
        if ratio is None:
            vars()['split_'+name] = vars()['fns_'+name]
        else:
            fns_split += vars()['fns_'+name]
            
    ## random split
    random.seed(seed)
    random.shuffle(fns_split)
    n_start = 0
    for name, ratio in split_ratio.items():
        if ratio is None:
            continue
        n = math.floor(len(fns_split)*ratio)
        vars()['split_'+name] = fns_split[n_start:n_start+n]
        n_start += n
    random.seed(time.time())
    return {'train': vars()['split_train'], 'val': vars()['split_val'], 'test': vars()['split_test']}


def split_data_reg(split_ratio, data, seed):

    ## shuffle data
    np.random.seed(seed)
    random.shuffle(data)
    np.random.seed(int(time.time()))
    
    ## split data
    ratio_list = [(name, ratio) for name, ratio in split_ratio.items()]
    name_list = [name for name, _ in ratio_list]
    n_list = [math.floor(len(data)*ratio) for _, ratio in ratio_list[:-1]]
    n_list = n_list + [len(data) - np.sum(n_list)]

    data_split = np.split(data, np.cumsum(n_list))[:-1]
    return {n: v for n, v in zip(name_list, data_split)}

        
# def split_data(data_fns, val_ratio, test_ratio, seed):
#     n_data = len(data_fn)
#     n_val = round(n_data*val_ratio)
#     n_test = round(n_data*test_ratio)
#     n_train = n_data - n_val - n_test

#     np.random.seed(seed)
#     idx_rnd = np.random.permutation(n_data)
#     train_fns = data_fns[idx_rnd[:n_train]]
#     val_fns = data_fns[idx_rnd[n_train:n_train+n_val]]
#     test_fns = data_fns[idx_rnd[n_train+n_val:n_train+n_val+n_test]]
#     return train_fns, val_fns, test_fns
    
    

def init_loader_reg(dataset_fn, data_split, tforms, tforms_y, rnd, batch_size, num_workers):
    dataset = dataset_fn(data_split, transform_x=transforms.Compose(tforms), transform_y=transforms.Compose(tforms_y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=rnd, num_workers=num_workers)
    return loader





    

        



class DetectionListDataset:
    def __init__(self, split, transform=None, target_transform=None, loader=default_loader, domain_label=None):
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = [(fn, label) for fn, label in zip(split['fn'], split['label'])]
        self.domain_label = domain_label
        
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = target if self.domain_label is None else self.domain_label
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
    


class DetectionData:
    def __init__(self, root, batch_size,
                 dataset_fn,
                 data_split, 
                 #split_ratio,
                 sample_size,
                 domain_label,
                 train_rnd, val_rnd, test_rnd,
                 train_aug, val_aug, test_aug,
                 aug_types,
                 num_workers,
                 tforms_dft, tforms_dft_rnd,
                 collate_fn=None,
                 #ext,
                 #seed,
    ):
        ## data augmentation tforms
        tforms_aug = get_aug_tforms(aug_types)

        ## tforms for each data split
        tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
        tforms_train = tforms_train + tforms_aug if train_aug else tforms_train
        tforms_val = tforms_dft_rnd if val_rnd else tforms_dft
        tforms_val = tforms_val + tforms_aug if val_aug else tforms_val
        tforms_test = tforms_dft_rnd if test_rnd else tforms_dft
        tforms_test = tforms_test + tforms_aug if test_aug else tforms_test

        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)
        
        # ## splits
        # split_list = get_split_list(split_ratio, root, ext, seed)
        # classes, class_to_idx = find_classes(root)

        ## truncate samples
        for name, value in data_split.items():
            if sample_size[name] is None:
                continue
            data_split[name] = {k: v[:sample_size[name]] for k, v in value.items()}

        ## create loaders
        dataset = dataset_fn(data_split['train'], transform=transforms.Compose(tforms_train), domain_label=domain_label)
        self.train = DataLoader(dataset, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers, collate_fn=collate_fn)

        dataset = dataset_fn(data_split['val'], transform=transforms.Compose(tforms_val), domain_label=domain_label)
        self.val = DataLoader(dataset, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers, collate_fn=collate_fn)

        dataset = dataset_fn(data_split['test'], transform=transforms.Compose(tforms_test), domain_label=domain_label)
        self.test = DataLoader(dataset, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers, collate_fn=collate_fn)


        
# class Data_old:
#     def __init__(self, root, batch_size,
#                  dataset_fn, 
#                  train_rnd, val_rnd, test_rnd,
#                  train_aug, val_aug, test_aug,
#                  aug_types,
#                  num_workers,
#                  tforms_dft, tforms_dft_rnd,
#                  seed=0,
#     ):
#         ## data augmentation tforms
#         tforms_aug = get_aug_tforms(aug_types)
        
#         ## tforms for each data split
#         tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
#         tforms_train += tforms_aug if train_aug else []
#         tforms_val = tforms_dft if val_rnd else tforms_dft
#         tforms_val += tforms_aug if val_aug else []
#         tforms_test = tforms_dft if test_rnd else tforms_dft
#         tforms_test += tforms_aug if test_aug else []

#         ## create loaders
#         subroot = os.path.join(root, "train")
#         if os.path.exists(subroot):
#             dataset = dataset_fn(subroot, transform=tforms.Compose(tforms_train))
#             self.train = DataLoader(dataset, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers)
#             shuffle_initial_data_order(self.train, seed)
#         else:
#             self.train = None
            
#         subroot = os.path.join(root, "val")
#         if os.path.exists(subroot):
#             dataset = dataset_fn(subroot, transform=tforms.Compose(tforms_val))
#             self.val = DataLoader(dataset, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers)
#             shuffle_initial_data_order(self.val, seed)
#         else:
#             self.val = None

#         subroot = os.path.join(root, "test")
#         if os.path.exists(subroot):
#             dataset = dataset_fn(subroot, transform=tforms.Compose(tforms_test))
#             self.test = DataLoader(dataset, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers)
#             shuffle_initial_data_order(self.test, seed)
#         else:
#             self.test = None


class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform, domain_label=None):
        super().__init__(root, transform=transform)
        self.domain_label = domain_label


    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        target = target if self.domain_label is None else self.domain_label
        return sample, target
    
            
class ImageData:
    def __init__(self, root, batch_size,
                 train_rnd, val_rnd, test_rnd,
                 train_aug, val_aug, test_aug,
                 aug_types,
                 num_workers,
                 tforms_dft, tforms_dft_rnd,
                 domain_label=None,
                 seed=0, 
    ):
        ## data augmentation tforms
        tforms_aug = get_aug_tforms(aug_types)
        
        ## tforms for each data split
        tforms_train = tforms_dft_rnd if train_rnd else tforms_dft
        tforms_train += tforms_aug if train_aug else []
        tforms_val = tforms_dft if val_rnd else tforms_dft
        tforms_val += tforms_aug if val_aug else []
        tforms_test = tforms_dft if test_rnd else tforms_dft
        tforms_test += tforms_aug if test_aug else []

        ## create loaders
        #dataset = datasets.ImageFolder(os.path.join(root, "train"), transform=transforms.Compose(tforms_train))
        dataset = ImageDataset(os.path.join(root, "train"), transform=transforms.Compose(tforms_train), domain_label=domain_label)
        self.train = DataLoader(dataset, batch_size=batch_size, shuffle=train_rnd, num_workers=num_workers)
        #dataset = datasets.ImageFolder(os.path.join(root, "val"), transform=transforms.Compose(tforms_val))
        dataset = ImageDataset(os.path.join(root, "val"), transform=transforms.Compose(tforms_val), domain_label=domain_label)
        self.val = DataLoader(dataset, batch_size=batch_size, shuffle=val_rnd, num_workers=num_workers)
        #dataset = datasets.ImageFolder(os.path.join(root, "test"), transform=transforms.Compose(tforms_test))
        dataset = ImageDataset(os.path.join(root, "test"), transform=transforms.Compose(tforms_test), domain_label=domain_label)
        self.test = DataLoader(dataset, batch_size=batch_size, shuffle=test_rnd, num_workers=num_workers)

        ## shuffle initial order
        shuffle_initial_data_order(self.train, seed)
        shuffle_initial_data_order(self.val, seed)
        shuffle_initial_data_order(self.test, seed)

        
##
## regression
##
class RegressionDatasetLight(Dataset):
    def __init__(self, data, transform_x, transform_y, label_index=-1):
        self.label_index = label_index
        self.data = data
        self.transform_x = transform_x
        self.transform_y = transform_y
        
            
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        data_i = self.data[idx]
        y = [data_i[self.label_index]]
        x = np.delete(data_i, self.label_index)
        return self.transform_x(x), self.transform_y(y)

        
class RegressionDataset(Dataset):
    def __init__(self, root, singlefile=False, label_index=-1):
        self.singlefile = singlefile
        self.label_index = label_index
        if self.singlefile:
            fn = glob.glob(os.path.join(root, "*.pk"))[0]
            self.data = pickle.load(open(fn, 'rb'))
        else:
            self.fns = glob.glob(os.path.join(root, "*.pk"))

            
    def __len__(self):
        if self.singlefile:
            return len(self.fns)
        else:
            return len(self.data)

        
    def __getitem__(self, idx):
        if self.singlefile:
            with open(self.fns[idx], "rb") as f:
                return pickle.load(f)
        else:
            data_i = data[idx]
            y = data[self.label_index]
            x = np.delete(data, self.label_index)
            return x, y

        
class RegressionDataLight:
    def __init__(self, root, batch_size,
                 dataset_fn,
                 split_ratio,
                 sample_size,
                 #domain_label,
                 train_rnd, val_rnd, test_rnd,
                 train_aug, val_aug, test_aug,
                 aug_types,
                 num_workers,
                 tforms_x_dft, tforms_x_dft_rnd,
                 tforms_y_dft, tforms_y_dft_rnd,
                 #ext,
                 seed,
    ):
        ## data augmentation tforms
        tforms_aug = get_aug_tforms(aug_types)

        ## tforms for each data split
        tforms_train = tforms_x_dft_rnd if train_rnd else tforms_x_dft
        tforms_train = tforms_train + tforms_aug if train_aug else tforms_train
        tforms_val = tforms_x_dft_rnd if val_rnd else tforms_x_dft
        tforms_val = tforms_val + tforms_aug if val_aug else tforms_val
        tforms_test = tforms_x_dft_rnd if test_rnd else tforms_x_dft
        tforms_test = tforms_test + tforms_aug if test_aug else tforms_test

        tforms_y_train = tforms_y_dft_rnd if train_rnd else tforms_y_dft
        tforms_y_val= tforms_y_dft_rnd if val_rnd else tforms_y_dft
        tforms_y_test = tforms_y_dft_rnd if test_rnd else tforms_y_dft

        print("[tforms_train] ", tforms_train)
        print("[tforms_val] ", tforms_val)
        print("[tforms_test] ", tforms_test)

        ## load data
        fn = glob.glob(os.path.join(root, "*.pk"))[0]
        data = pickle.load(open(fn, 'rb'))
        
        ## splits
        data_split = split_data_reg(split_ratio, data, seed)

        ## truncate samples
        for name, value in data_split.items():
            if sample_size[name] is None:
                continue
            data_split[name] = value[:sample_size[name]]

        ## create loaders
        self.train = init_loader_reg(dataset_fn, data_split['train'], tforms_train, tforms_y_train, train_rnd, batch_size, num_workers)
        self.val = init_loader_reg(dataset_fn, data_split['val'], tforms_val, tforms_y_val, val_rnd, batch_size, num_workers)
        self.test = init_loader_reg(dataset_fn, data_split['test'], tforms_test, tforms_y_test, test_rnd, batch_size, num_workers)


