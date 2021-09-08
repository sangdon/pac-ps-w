import os, sys
import numpy as np
import argparse
import pickle
import time

import torch
import torch as tc
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from advertorch.attacks import LinfPGDAttack
import torchvision.models as models

IMG_MEAN=[0.485, 0.456, 0.406]
IMG_STD=[0.229, 0.224, 0.225]

class ExampleNormalizer(nn.Module):
    def __init__(self, mdl, mean=IMG_MEAN, std=IMG_STD):
        super().__init__()
        self.mdl = mdl
        self.mean = mean
        self.std = std

        
    def forward(self, x):
        x = TF.normalize(tensor=x, mean=self.mean, std=self.std)
        x = self.mdl(x)
        return x
    

# def init_imagenet_loaders(args):

#     traindir = os.path.join(args.data, 'train')
#     valdir = os.path.join(args.data, 'val')
#     testdir = os.path.join(args.data, 'test')

#     normalize = transforms.Normalize(mean=IMG_MEAN,
#                                      std=IMG_STD)

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             #transforms.RandomResizedCrop(224),
#             #transforms.RandomHorizontalFlip(),
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             #normalize,
#         ]))

#     if hasattr(args, 'distributed') and args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#         num_workers=args.workers, pin_memory=True, sampler=train_sampler)

#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             #normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=False, ## shuffle should be False for model composition
#         num_workers=args.workers, pin_memory=True)

#     test_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(testdir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             #normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)

#     return train_loader, val_loader, test_loader


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target

    
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (_, images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if hasattr(args, 'adv') and args.adv:
                images = adversary.perturb(images, target)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i+1 % args.print_freq == 0:
                progress.display(i)
    # TODO: this should also be done with the ProgressMeter
    if args.print_freq is not np.inf:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def get_custom_imagenet_test_loader(args):

    ## train
    train_loader = tc.utils.data.DataLoader(
        MyImageFolder(os.path.join(args.data, 'train'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ## val
    val_loader = tc.utils.data.DataLoader(
        MyImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ## test
    test_loader = tc.utils.data.DataLoader(
        MyImageFolder(os.path.join(args.data, 'test'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def generate(args):
    #os.makedirs(args.results_root, exist_ok=True)
    
    ## load the pretrained model
    model = models.__dict__[args.arch](pretrained=True)
    model = ExampleNormalizer(model)
    assert(not args.gpu)
    model = tc.nn.DataParallel(model)
    model.eval()
    model.cuda()
    
    ## init data loader
    train_loader, val_loader, test_loader = get_custom_imagenet_test_loader(args)

    ## measure accuracy
    #_, _, test_loader_original = init_imagenet_loaders(args)
    acc_top1 = validate(test_loader, model, tc.nn.CrossEntropyLoss(), args)
    print('# top1 accuracy = %f%%'%(acc_top1))


    ## init an adverary
    assert(args.adv_norm == 'Linf')
    adversary = LinfPGDAttack(
        model, loss_fn=tc.nn.CrossEntropyLoss(reduction="sum"), eps=args.adv_eps,
        nb_iter=args.adv_nb_iter, eps_iter=args.adv_eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=args.adv_targeted)
    
    ## generate adversarial examples
    n = 0.0
    for loader, split_type in zip([test_loader, val_loader, train_loader], ['test', 'val', 'train']):
        for path, x, y in loader:

            i_filtered = []
            for i, path_i in enumerate(path):
                print(path_i)
                splits = path_i.split('/')
                label_name = splits[2]
                file_name = splits[3].split('.')[0] + '.advex'
                path_i_new = os.path.join(args.results_root, split_type, label_name, file_name)
                if not os.path.exists(path_i_new):
                    i_filtered.append(i)
                    
            if len(i_filtered) == 0:
                continue
            path = [path[i] for i in i_filtered]
            x = x[i_filtered]
            y = y[i_filtered]

            t_start = time.time()
            x = x.to(args.device)
            y = y.to(args.device)
            xp = adversary.perturb(x, y)
            assert(tc.norm(x.flatten()-xp.flatten(), p=np.inf) <= args.adv_eps + 1e-6) ##this takes some time, but fine
            n += x.shape[0]

            ## save images
            xp = xp.cpu().numpy()
            for path_i, xp_i in zip(path, xp):
                splits = path_i.split('/')
                label_name = splits[2]
                file_name = splits[3].split('.')[0] + '.advex'
                path_i_new = os.path.join(args.results_root, split_type, label_name, file_name)
                print(path_i_new)

                xp_i = (xp_i*255).astype(np.uint8)
                os.makedirs(os.path.dirname(path_i_new), exist_ok=True)
                pickle.dump(xp_i, open(path_i_new, 'wb'))

            print('[#processed = %d, %f sec./image]'%(n, (time.time() - t_start)/x.shape[0]))
        
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial imagenet examples')
    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('-a', '--arch', default='resnet101')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-p', '--print-freq', type=int, default=np.inf, help='print frequency (default: inf)')
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training. ')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--results_root', type=str, default='imagenet_advex_resnet101')    
    
    parser.add_argument('--adv_norm', type=str, default='Linf', help='Attack type') #"Linf" "L1" "L2"
    parser.add_argument('--adv_eps', type=float, default=0.1, help='Max distance of attack')
    parser.add_argument('--adv_nb_iter', type=int, default=50, help='Attack iterations')
    parser.add_argument('--adv_eps_iter', type=float, default=0.01, help='Iteration step size')
    parser.add_argument('--adv_targeted', action='store_true')
    
    args = parser.parse_args()
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:%d'%(args.gpu))
    args.results_root += f'_eps_{args.adv_eps}'
    generate(args)




