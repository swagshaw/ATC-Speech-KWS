"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午10:25
@Author  : Yang "Jan" Xiao 
@Description : bcresnet
"""
import torch
from torch import Tensor
import torch.nn as nn
from torchaudio.transforms import MFCC


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C * S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(self, n_class: int, scale: int):
        super(BCResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16*scale, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16*scale, 8*scale)
        self.block1_2 = BroadcastedBlock(8*scale)

        self.block2_1 = TransitionBlock(8*scale, 12*scale, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12*scale, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12*scale, 16*scale, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16*scale, 20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20*scale, 20*scale, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20*scale, 32*scale, 1, bias=False)
        self.conv4 = nn.Conv2d(32*scale, n_class, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1_1(out)
        out = self.block1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)
        out = self.conv4(out).squeeze()
        return out


class MFCC_BCResnet(nn.Module):
    def __init__(self, bins: int, channel_scale: int, num_classes=12):
        super(MFCC_BCResnet, self).__init__()
        self.sampling_rate = 16000
        self.bins = bins
        self.channel_scale = channel_scale
        self.num_classes = num_classes
        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.bins, log_mels=True, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64})
        self.bc_resnet = BCResNet(num_classes, channel_scale)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits = self.bc_resnet(mel_sepctogram)
        return logits


########################################################################################################################
#                                                Squeeze and Excitation                                                #
########################################################################################################################
class SELayer(nn.Module):
    def __init__(self, dim, reduction=16, attend_dim="chan"):
        super(SELayer, self).__init__()
        self.attend_dim = attend_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hid_dim = dim // reduction

        if hid_dim < 4:
            hid_dim = 4

        if attend_dim == "chan-freq":
            self.fc = nn.Sequential(nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=False),
                                    nn.Sigmoid())

        else:
            self.fc = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
                                    # nn.BatchNorm1d(hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hid_dim, dim, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):   #x size : [bs, chan, frames, freqs]
        b, c, t, f = x.size()
        if self.attend_dim == "chan":
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

        elif self.attend_dim == "chan_timewise":
            y = torch.mean(x, dim=3).transpose(1, 2)  #x size : [bs, frames, chan]
            y = self.fc(y).transpose(1, 2).view(b, c, t, 1)

        elif self.attend_dim == "freq":
            y = torch.mean(x, dim=(1, 2))
            y = self.fc(y).view(b, 1, 1, f)

        elif self.attend_dim == "freq_timewise":
            y = torch.mean(x, dim=1)                  #x size : [bs, frames, freqs]
            y = self.fc(y).view(b, 1, t, f)

        elif self.attend_dim == "chan-freq":
            y = torch.mean(x, dim=2).view(b, 1, c, f)
            y = self.fc(y).view(b, c, 1, f)

        return x * y.expand_as(x)
    
class SEResNet(torch.nn.Module):
    def __init__(self, n_class: int, scale: int):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16*scale, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16*scale, 8*scale)
        self.block1_2 = BroadcastedBlock(8*scale)
        self.se1_1 = SELayer(8*scale, reduction=4, attend_dim="chan")
        self.se1_2 = SELayer(8*scale, reduction=4, attend_dim="'freq_timewise'")
        self.block2_1 = TransitionBlock(8*scale, 12*scale, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12*scale, dilation=(1, 2), temp_pad=(0, 2))
        self.se2_1 = SELayer(12*scale, reduction=4, attend_dim="chan")
        self.se2_2 = SELayer(12*scale, reduction=4, attend_dim="'freq_timewise'")
        self.block3_1 = TransitionBlock(12*scale, 16*scale, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16*scale, dilation=(1, 4), temp_pad=(0, 4))
        self.se3_1 = SELayer(16*scale, reduction=4, attend_dim="chan")
        self.se3_2 = SELayer(16*scale, reduction=4, attend_dim="'freq_timewise'")
        self.block4_1 = TransitionBlock(16*scale, 20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20*scale, dilation=(1, 8), temp_pad=(0, 8))
        self.se4_1 = SELayer(20*scale, reduction=4, attend_dim="chan")
        self.se4_2 = SELayer(20*scale, reduction=4, attend_dim="'freq_timewise'")

        self.conv2 = nn.Conv2d(20*scale, 20*scale, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20*scale, 32*scale, 1, bias=False)
        self.conv4 = nn.Conv2d(32*scale, n_class, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.se1_1(out)
        out = self.se1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.se2_1(out)
        out = self.se2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)
        out = self.se3_1(out)
        out = self.se3_2(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)
        out = self.se4_1(out)
        out = self.se4_2(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)
        out = self.conv4(out).squeeze()
        return out
    
if __name__ == "__main__":
    x = torch.ones(128, 1, 16000)
    bcresnet = MFCC_BCResnet(bins=40, channel_scale=1, num_classes=30)
    _ = bcresnet(x)
    print('num parameters:', sum(p.numel() for p in bcresnet.parameters() if p.requires_grad))
