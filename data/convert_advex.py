import os, sys
import pickle
import glob
import numpy as np
import torch as tc
from torch import nn
import torchvision.models as models
import torchvision.transforms.functional as TF

import data

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

    
if __name__ == '__main__':
    root = 'data/imagenetadvex'
    fns = glob.glob(os.path.join(root, '**', '**', '*.advex'))
    for fn in fns:
        print(fn)
        x = pickle.load(open(fn, 'rb'))
        if x.dtype == 'float32':
            print(x.min(), x.max(), x.mean())
            x = (x * 255).astype(np.uint8)
            pickle.dump(x, open(fn, 'wb'))


    ## eval the model and dataset
    dsld = data.ImageNetAdvEx(root, 100, sample_size={'train': None, 'val': None, 'test': None})

    model = models.resnet101(pretrained=True)
    model = ExampleNormalizer(model)
    model = tc.nn.DataParallel(model)
    model.eval()
    model.cuda()


    def validate(mdl, ld):
        n = 0.0
        error = 0.0
        for x, y in ld:
            x, y = x.cuda(), y.cuda()
            yh = mdl(x).argmax(1)
            n += x.shape[0]
            error += (y != yh).sum().float()
        return error / n
    
    print('[train] top1 adv error = %f%%'%(validate(model, dsld.train)*100.0))
    print('[val] top1 adv error = %f%%'%(validate(model, dsld.val)*100.0))
    print('[val] top1 adv error = %f%%'%(validate(model, dsld.test)*100.0))
    
    
    
