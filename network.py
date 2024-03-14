#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vit_pytorch import ViT
from mlp_mixer_pytorch import MLPMixer
from math import ceil

relu_slp = 0.2
class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=relu_slp, dropout=None, groups = 1):
        super(Conv2DwithBN,self).__init__()
        
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups)]
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


class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=64, dim2=128, dim3=256, dim4=512, dim5=2048, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(64, dim1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 3), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 3), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 3), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2, padding = (1,1))
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(4, 4), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=2, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=5, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=5, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        nx = 214
        tr = 100
        [X,Y] = torch.meshgrid(torch.arange(nx),torch.arange(nx))
        self.mask = nn.Parameter((X-(nx-1)/2)**2 + (Y-(nx-1)/2)**2 <= tr**2, requires_grad = False)
        
        
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 64, 360, 64)
        x = self.convblock2_1(x) # (None, 128, 180, 64)
        x = self.convblock2_2(x) # (None, 128, 180, 64)
        x = self.convblock3_1(x) # (None, 128, 90, 64)
        x = self.convblock3_2(x) # (None, 128, 90, 64)
        x = self.convblock4_1(x) # (None, 256, 45, 64)
        x = self.convblock4_2(x) # (None, 256, 45, 64)
        x = self.convblock5_1(x) # (None, 256, 23, 32) 
        x = self.convblock5_2(x) # (None, 256, 23, 32)
        x = self.convblock6_1(x) # (None, 512, 12, 16)
        x = self.convblock6_2(x) # (None, 512, 12, 16)
        x = self.convblock7_1(x) # (None, 512, 8, 8)
        x = self.convblock7_2(x) # (None, 512, 8, 8)
        x = self.convblock8(x) # (None, 2048, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 2048, 2, 2)
        x = self.deconv1_2(x) # (None, 2048, 2, 2)
        x = self.deconv2_1(x) # (None, 512, 4, 4)
        x = self.deconv2_2(x) # (None, 512, 4, 4)
        x = self.deconv3_1(x) # (None, 256, 8, 8)
        x = self.deconv3_2(x) # (None, 256, 8, 8)
        x = self.deconv4_1(x) # (None,128, 40, 40)
        x = self.deconv4_2(x) # (None, 128, 40, 40)
        x = self.deconv5_1(x) # (None, 64, 240, 240)
        x = self.deconv5_2(x) # (None, 64, 240, 240)
        x = F.pad(x, [7, 7, 7, 7], mode="constant", value=0) # (None, 32, 214, 214)
        x = self.deconv6(x) # (None, 1, 214, 214)'
        x = torch.einsum('ijkl,kl->ijkl', x, self.mask)
        

        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256):
        super(Discriminator, self).__init__()
        self.convblock1_1 = Conv2DwithBN(1, dim1, stride=2)
        self.convblock1_2 = Conv2DwithBN(dim1, dim1)
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, stride=2)
        self.convblock2_2 = Conv2DwithBN(dim2, dim2)
        self.convblock3_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock3_2 = Conv2DwithBN(dim3, dim3)
        self.convblock4_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock4_2 = Conv2DwithBN(dim4, dim4)
        self.convblock5 = Conv2DwithBN(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x

class FCN4_Deep_Resize_Enc(nn.Module):
    def __init__(self, dim1=64, dim2=128, dim3=256, dim4=512, dim5=2048, ratio=1.0, upsample_mode='nearest', init = 'gauss', train_encoder = False):
        super(FCN4_Deep_Resize_Enc, self).__init__()

        if init == 'gauss':
              E = torch.randn((16,64))
        else:
              E = torch.zeros((16,64))
              for j in range(16):
                  E[j,4*j:4*(j+1)] = 1
        self.encoder = nn.Parameter(E, requires_grad = train_encoder)
        self.convblock1 = Conv2DwithBN(16, dim1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 3), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 3), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 3), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2, padding = (1,1))
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(4, 4), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=2, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=6, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=4, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        nx = 214
        tr = 92
        [X,Y] = torch.meshgrid(torch.arange(nx),torch.arange(nx))
        self.mask = nn.Parameter((X-(nx-1)/2)**2 + (Y-(nx-1)/2)**2 <= tr**2, requires_grad = False)
           
    def forward(self,x):
        # Encoder Part
        # Reshape and transpose self.encoder
        # encoder_reshaped = self.encoder.view(1, 16, 1, 1, 64).permute(0, 1, 4, 2, 3)
        # x = torch.einsum('aijkl, abcd -> abcd', encoder_reshaped, x)
        # print(x.shape)
        x = torch.einsum('ij, ajbc -> aibc', self.encoder, x)
        x = self.convblock1(x) # (None, 64, 360, 64)
        x = self.convblock2_1(x) # (None, 128, 180, 64)
        x = self.convblock2_2(x) # (None, 128, 180, 64)
        x = self.convblock3_1(x) # (None, 128, 90, 64)
        x = self.convblock3_2(x) # (None, 128, 90, 64)
        x = self.convblock4_1(x) # (None, 256, 45, 64)
        x = self.convblock4_2(x) # (None, 256, 45, 64)
        x = self.convblock5_1(x) # (None, 256, 23, 32) 
        x = self.convblock5_2(x) # (None, 256, 23, 32)
        x = self.convblock6_1(x) # (None, 512, 12, 16)
        x = self.convblock6_2(x) # (None, 512, 12, 16)
        x = self.convblock7_1(x) # (None, 512, 8, 8)
        x = self.convblock7_2(x) # (None, 512, 8, 8)
        x = self.convblock8(x) # (None, 2048, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 2048, 2, 2)
        x = self.deconv1_2(x) # (None, 2048, 2, 2)
        x = self.deconv2_1(x) # (None, 512, 4, 4)
        x = self.deconv2_2(x) # (None, 512, 4, 4)
        x = self.deconv3_1(x) # (None, 256, 8, 8)
        x = self.deconv3_2(x) # (None, 256, 8, 8)
        x = self.deconv4_1(x) # (None,128, 40, 40)
        x = self.deconv4_2(x) # (None, 128, 40, 40)
        x = self.deconv5_1(x) # (None, 64, 240, 240)
        x = self.deconv5_2(x) # (None, 64, 240, 240)
        x = F.pad(x, [11, 11, 11, 11], mode="constant", value=0) # (None, 32, 214, 214)
        x = self.deconv6(x) # (None, 1, 214, 214)'
        x = torch.einsum('ijkl,kl->ijkl', x, self.mask)
        
        return x


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None):
        super(Conv_HPGNN, self).__init__()
        layers = [
            Conv2DwithBN(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            Conv2DwithBN(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            Conv2DwithBN(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            Conv2DwithBN(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HPGNN(nn.Module):
    def __init__(self, dim0=32, dim1=64, dim2=128, dim3=256, dim4=512, dim5=1024):
        super(HPGNN, self).__init__()
        self.convblock1 = nn.Conv2d(32, dim0, kernel_size=1, stride=1, padding=0)
        self.convblock2 = Conv_HPGNN(dim0, dim1, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock3 = Conv_HPGNN(dim1, dim2, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock4 = Conv_HPGNN(dim2, dim3, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock5 = Conv_HPGNN(dim3, dim4, kernel_size=(2, 1), stride=(2, 1), padding=(1,0))
        self.convblock6 = Conv_HPGNN(dim4, dim5)
        self.deconv1 = Deconv_HPGNN(dim5, dim4, 16)
        self.deconv2 = Deconv_HPGNN(dim4, dim3, 8)
        self.deconv3 = Deconv_HPGNN(dim3, dim2, 8)
        self.deconv4 = Deconv_HPGNN(dim2, dim1, 4)
        self.conv_last = nn.Conv2d(dim1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv_last(x)
        return torch.clamp(x, -1.0, 1.0)


model_dict = {
    'FCN4_Deep_Resize_2': FCN4_Deep_Resize_2,
    'FCN4_Deep_Resize_Enc': FCN4_Deep_Resize_Enc,
    'HPGNN':HPGNN,
}

if __name__ == '__main__':
    device = torch.device('cpu')
    # model = MLP_Mixer_Decoder_Resize() # 35594827
    # model = MLP_Mixer_Decoder()        # 41007691
    # model = ViT_Decoder_Resize()       # 63177315
    # model = ViT_Decoder()              # 68590179
    # model = FCN4_Salt_Resize()         # 11737379
    # model = FCN4_Salt()                # 13742371
    ratio = 0.1
    model = FCN4_Deep_Resize_2(ratio=ratio)       # 18996259
    #model = HPGNN()
    # model = FCN4_V2S_Deep_2()          # 83531903, 65706111
    # model = FCN4_V2S_Deep()            # 22972031
    # model = SeisT_Decoder()            # 68590179
    # model = SeisT_Decoder_Resize()     # 63177315
    # model = FCN4_No_BN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    # x= torch.rand((3, 9, 401, 301))
    # x = torch.rand((3, 5, 600, 60))
    # x = torch.rand((3, 1, 70, 70))
    x = torch.rand((3, 32, 360, 32))
    y = model(x)
    print(y.shape)

