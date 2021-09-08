import sys, os
import torch as tc

def decode_input(x):
    if type(x) is tuple:
        ## assume (img, label) tupble
        img = x[0]
        label = x[1]
    else:
        img = x
        label = None
    return img, label


class GaussianNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, img):
        img, label = decode_input(img)
        if self.std > 0:
            img = img + tc.normal(0, self.std, size=img.shape)
        return (img, label)

    
class LabelDepGaussianNoise:
    def __init__(self, std=0.0):
        self.std = std

        
    def __call__(self, img):
        _, label = decode_input(img)
        if label%2 == 1:
            # odd case
            std = self.std
        else:
            std = 0.0
        return GaussianNoise(std)(img)
    
    
    
class IntensityScaling:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        

    def __call__(self, img):
        img, label = decode_input(img)
        img = img * (self.low + tc.rand(1)*(self.high - self.low))        
        return (img, label)

    
class LabelDepIntensityScaling:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        

    def __call__(self, img):
        _, label = decode_input(img)
        if label%2 == 1:
            # odd case
            low, high = self.low, self.high
        else:
            low, high = 1.0, 1.0
        return IntensityScaling(low, high)(img)
    

class Clamp:
    def __init__(self, mn=0.0, mx=1.0):
        self.mn = mn
        self.mx = mx

    def __call__(self, img):
        img, label = decode_input(img)
        img = tc.clamp(img, self.mn, self.mx)
        return (img, label)

                
def get_aug_tforms(aug_names):
    if aug_names is None:
        return []
    aug_tforms = []
    for aug_name in aug_names:
        if 'noise' in aug_name:
            std = float(aug_name.split(":")[1])
            aug_tforms += [GaussianNoise(std)]
        elif 'intensityscaling' in aug_name:
            width = float(aug_name.split(":")[1])
            aug_tforms += [IntensityScaling(width)]
        elif 'clamp' in aug_name:
            mn = float(aug_name.split(":")[1])
            mx = float(aug_name.split(":")[2])
            aug_tforms += [Clamp(mn, mx)]
            
        elif aug_name == 'svhnspecific':
            raise NotImplementedError
            aug_tforms += [
                GaussianNoise(0.1),
            ]
        elif 'LDGN' in aug_name:
            std = float(aug_name.split(":")[1])
            aug_tforms += [LabelDepGaussianNoise(std)]
        elif 'LDIS' in aug_name:
            low = float(aug_name.split(":")[1])
            high = float(aug_name.split(":")[2])
            aug_tforms += [LabelDepIntensityScaling(low, high)]
        else:
            raise NotImplementedError

    return aug_tforms
