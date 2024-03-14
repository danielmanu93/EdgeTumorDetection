#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vit_pytorch import ViT
from mlp_mixer_pytorch import MLPMixer
from math import ceil

relu_slp = 0.1
class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=relu_slp, dropout=None, zero_in = False):
        super(Conv2DwithBN,self).__init__()
        if not zero_in:
            layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            l = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
            weights = torch.zeros_like(l.weight)
            with torch.no_grad():
                 l.weight = nn.Parameter(weights)
            layers = [l]
          
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        #layers.append(nn.LeakyReLU(relu_slop, inplace=True)) 0.2
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        out = self.Conv2DwithBN(x)
        return out


class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, zero_in = True):
        super(Conv2DwithBN_Tanh, self).__init__()
        if not zero_in:
            layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            l = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
            weights = torch.zeros_like(l.weight)
            with torch.no_grad():
                 l.weight = nn.Parameter(weights)
            layers = [l]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)


class Deconv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(Deconv2DwithBN, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        #layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.LeakyReLU(relu_slp, inplace=True))
        self.Deconv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Deconv2DwithBN(x)


class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slp, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)


class TumorClassifier(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, ratio=1.0, upsample_mode='nearest'):
        super(TumorClassifier, self).__init__()
        self.convblock1_0 = Conv2DwithBN(1, dim1, kernel_size=5, stride=1, padding=0)
        self.convblock1_1 = Conv2DwithBN(dim1, dim1, kernel_size=3, stride=1, padding=0)

        self.mpool = nn.MaxPool2d(2,stride = 2)

        self.convblock2_0 = Conv2DwithBN(dim1, dim2, kernel_size=3, stride=1, padding=0)
        self.convblock2_1 = Conv2DwithBN(dim2, dim2, kernel_size=3, stride=1, padding=0)
        #self.mpool
	
        self.convblock3_0 = Conv2DwithBN(dim2, dim3, kernel_size=3, stride=1, padding=1)
        self.convblock3_1 = Conv2DwithBN(dim3, dim3, kernel_size=3, stride=1, padding=1)
        #self.mpool

        self.convblock4_0 = Conv2DwithBN(dim3, dim4, kernel_size=3, stride=1, padding=1)
        self.convblock4_1 = Conv2DwithBN(dim4, dim4, kernel_size=3, stride=1, padding=1)
        #self.mpool

        self.deconv3 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.convblock3_2 = Conv2DwithBN(2*dim3, dim3, kernel_size=3, stride=1, padding=1)
        self.convblock3_3 = Conv2DwithBN(dim3, dim3, kernel_size=3, stride=1, padding=1)

        self.deconv2 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.convblock2_2 = Conv2DwithBN(2*dim2, dim2, kernel_size=3, stride=1, padding=2)
        self.convblock2_3 = Conv2DwithBN(dim2, dim2, kernel_size=3, stride=1, padding=2)

        self.deconv1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.convblock1_2 = Conv2DwithBN(2*dim1, dim1, kernel_size=3, stride=1, padding=2)
        self.convblock1_3 = Conv2DwithBN(dim1, dim1, kernel_size=5, stride=1, padding=4)

        self.out = Conv2DwithBN_Tanh(dim1, 1)

    def expand(self, x):
        x1 = self.convblock1_0(x)
        x1 = self.convblock1_1(x1)

        x2 = self.mpool(x1)
        x2 = self.convblock2_0(x2)
        x2 = self.convblock2_1(x2)

        x3 = self.mpool(x2)
        x3 = self.convblock3_0(x3)
        x3 = self.convblock3_1(x3)

        x4 = self.mpool(x3)
        x4 = self.convblock4_0(x4)
        x4 = self.convblock4_1(x4)
        x4 = self.deconv3(x4)
        return x1, x2, x3, x4
        
        
    def forward(self,x):

        x1, x2, x3, x4 = self.expand(x)
        x3 = torch.cat((x3,x4), dim = 1)
        x3 = self.convblock3_2(x3)
        x3 = self.convblock3_3(x3)
        x3 = self.deconv2(x3)

        x2 = torch.cat((x2,x3), dim = 1)
        x2 = self.convblock2_2(x2)
        x2 = self.convblock2_3(x2)
        x2 = self.deconv1(x2)

        x1 = torch.cat((x1,x2), dim = 1)
        x1 = self.convblock1_2(x1)
        x1 = self.convblock1_3(x1)
        x1 = 0.5*self.out(x1)+0.5

        return x1


