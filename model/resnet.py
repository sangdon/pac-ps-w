import os, sys
import warnings

import torch as tc
from torch import nn
import torch.nn.functional as F
from torchvision import models
import threading
import re

class ResNet(nn.Module):
    def __init__(self, n_labels, resnet_id, path_pretrained=None):
        super().__init__()

        if path_pretrained is None:
            self.model = getattr(models, 'resnet%d'%(resnet_id))(num_classes=n_labels, pretrained=False)
        elif path_pretrained == 'pytorch':
            self.model = getattr(models, 'resnet%d'%(resnet_id))(num_classes=n_labels, pretrained=True)
        else:
            self.model = getattr(models, 'resnet%d'%(resnet_id))(num_classes=n_labels, pretrained=False)
            self.model.load_state_dict({k.replace('module.', '').replace('mdl.', '').replace('model.', ''): v for k, v in
                                        tc.load(path_pretrained, map_location=tc.device('cpu')).items()})

        self.feat = {}
        def feat_hook(model, input, output):
            #self.feat = tc.flatten(output, 1)
            self.feat[threading.get_ident()] = tc.flatten(output, 1)
            return output
        self.model.avgpool.register_forward_hook(feat_hook)
                

    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        
        x = self.model(x)
        
        return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': self.feat[threading.get_ident()]}
        #return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': self.feat}

    
class ResNetFeat(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.model = mdl.model.fc

    def forward(self, x, training=False):
        assert(training == False)
        self.eval()

        feat = x
        x = self.model(feat)        
        return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': feat}


def ResNet18(n_labels, path_pretrained=None):
    return ResNet(n_labels, 18, path_pretrained=path_pretrained)


def ResNet101(n_labels, path_pretrained=None):
    return ResNet(n_labels, 101, path_pretrained=path_pretrained)


def ResNet152(n_labels, path_pretrained=None):
    return ResNet(n_labels, 152, path_pretrained=path_pretrained)


