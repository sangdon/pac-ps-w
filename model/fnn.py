import os, sys
import warnings

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, n_in, n_out, n_hiddens, n_layers, path_pretrained=None):
        super().__init__()
        
        models = []
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            models.append(nn.Linear(n, n_hiddens))
            models.append(nn.ReLU())
            models.append(nn.Dropout(0.5))
        models.append(nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out))
        self.model = nn.Sequential(*models)

        if path_pretrained is not None:
            warnings.warn('use a unified model structure for model loading')
            self.model.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
                                        tc.load(path_pretrained, map_location=tc.device('cpu')).items()})
            # self.model.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
            #                             tc.load(path_pretrained, map_location=tc.device('cpu')).items()})

        
        
    def forward(self, x, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()
        logits = self.model(x)
        if logits.shape[1] == 1:
            probs = tc.sigmoid(logits)
        else:
            probs = F.softmax(logits, -1)
        return {'fh': logits, 'ph': probs, 'yh_top': logits.argmax(-1), 'ph_top': probs.max(-1)[0], 'feat': x}


class Linear(FNN):
    def __init__(self, n_in, n_out, n_hiddens=None, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=0, path_pretrained=path_pretrained)


class SmallFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1, path_pretrained=path_pretrained)

    
class MidFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2, path_pretrained=path_pretrained)

        
class BigFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4, path_pretrained=path_pretrained)






