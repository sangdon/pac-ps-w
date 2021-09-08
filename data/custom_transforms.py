import os, sys
import torch as tc
from torchvision import transforms as tforms
from data.aug import decode_input


class Identity:
    def __call__(self, x):
        return x

class CustomTransform:
    def __call__(self, img):
        img, label = decode_input(img)
        img = self.tf(img)
        return (img, label)
    
    
class ToJustTensor(CustomTransform):
    def __init__(self):
        self.tf = lambda x: tc.tensor(x)

        
class ToTensor(CustomTransform):
    def __init__(self):
        self.tf = tforms.ToTensor()

    
# class Normalizer:
#     def __init__(self, n):
#         self.n = tc.tensor(n)

        
#     def __call__(self, x):
#         img, label = decode_input(x)
#         img = img / self.n
#         return (img, label)

    
class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.tf = tforms.Normalize(mean, std)
        
    
class Grayscale(CustomTransform):
    def __init__(self, n_channels):
        self.tf = tforms.Grayscale(n_channels)
        

class Resize(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.Resize(size)

    
class CenterCrop(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.CenterCrop(size)

        
class RandomResizedCrop(CustomTransform):
    def __init__(self, size):
        self.tf = tforms.RandomResizedCrop(size)

        
class RandomHorizontalFlip(CustomTransform):
    def __init__(self):
        self.tf = tforms.RandomHorizontalFlip()
