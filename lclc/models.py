import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from lclc.netvlad import NetVLAD

class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)

import numpy as np
class dual_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_d = models.vgg16(weights=None)
        encoder_r = models.vgg16(weights=None)
        layers_d = list(encoder_d.features.children())[:-2]
        layers_r = list(encoder_r.features.children())[:-2]
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        for layer in layers_d[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in layers_r[:-5]:
            for p in layer.parameters():
                p.requires_grad = False

        self.encoder_d = nn.Sequential(*layers_d)
        self.encoder_r = nn.Sequential(*layers_r)
        self.enc_dim = 512

    def forward(self, x, x_isrange):
        idx_disp = ~x_isrange
        out_disp = self.encoder_d(x)
        out_disp = idx_disp[:, None, None, None] * out_disp

        idx_range = x_isrange
        out_range = self.encoder_r(x)
        out_range = idx_range[:, None, None, None] * out_range

        out = out_disp + out_range
        return out

