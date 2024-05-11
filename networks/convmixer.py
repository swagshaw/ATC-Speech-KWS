#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Oct 10, 2021, 17:23:06
# @author: dianwen ng
# @file  : ConvMixer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        elif self.attend_dim == "time":
            y = torch.mean(x, dim=(1, 3))
            y = self.fc(y).view(b, 1, t, 1)
        elif self.attend_dim == "freq_timewise":
            y = torch.mean(x, dim=1)                  #x size : [bs, frames, freqs]
            y = self.fc(y).view(b, 1, t, f)
        elif self.attend_dim == "time_freqwise":
            y = torch.mean(x, dim=3)                 #x size : [bs, chan, frames]
            y = self.fc(y).view(b, t, 1, f)
        elif self.attend_dim == "chan-freq":
            y = torch.mean(x, dim=2).view(b, 1, c, f)
            y = self.fc(y).view(b, c, 1, f)

        return x * y.expand_as(x)
    
class ECALayer(nn.Module):
    def __init__(self, dim, k_size=3, reduction=16, attend_dim="chan"):
        super(ECALayer, self).__init__()
        self.attend_dim = attend_dim
        self.k_size = k_size
        self.reduction = reduction
        hid_dim = max(dim // reduction, 4)  # Ensuring a minimum dimension size for stability

        if hid_dim < 4:
            hid_dim = 4

        if attend_dim == "chan":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        elif attend_dim == "chan-freq":
            # Example configuration for channel-frequency attention
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(1, 1, kernel_size=(k_size, k_size), padding=((k_size - 1) // 2, (k_size - 1) // 2), bias=False)
            self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        b, c, t, f = x.size()

        if self.attend_dim == "chan":
            y = self.avg_pool(x)  # Global average pooling
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
        elif self.attend_dim == "chan-freq":
            # Assume some reshaping and pooling adapted for channel-frequency
            y = self.avg_pool(x).view(b, 1, c, -1)  # Adjusted pooling
            y = self.conv(y).view(b, c, 1, 1)
            y = self.sigmoid(y)
        
        elif self.attend_dim == "chan_timewise":
            y = torch.mean(x, dim=3).transpose(1, 2)  #x size : [bs, frames, chan]
            y = self.fc(y).transpose(1, 2).view(b, c, t, 1)

        elif self.attend_dim == "freq":
            y = torch.mean(x, dim=(1, 2))
            y = self.fc(y).view(b, 1, 1, f)

        elif self.attend_dim == "freq_timewise":
            y = torch.mean(x, dim=1)                  #x size : [bs, frames, freqs]
            y = self.fc(y).view(b, 1, t, f)

        return x * y.expand_as(x)    
## activation layer
class Swish(nn.Module):
    """Swish is a smooth, non-monotonic function that 
    consistently matches or outperforms ReLU on 
    deep networks applied to a variety of challenging 
    domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()
                
## depthwise separable convolution (1D)
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=True):
    
        super(SeparableConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)
        
        if pointwise:
            self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pointwise(x)
        return x
    
    
## depthwise separable convolution (2D)    
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=False):
        
        super(SeparableConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)
        if pointwise:
            self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.pointwise(x)
        return x

    
## MLP layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """ Perform feed forward layer for mixer
        Parameter args:
            dim: in_channel dimension
            hidden_dim: intermediate dimension during FFN
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)
    
    
## mixer block  
class MixerBlock(nn.Module):
    def __init__(self, time_dim, freq_dim, dropout=0.):
        super(MixerBlock, self).__init__()

        self.time_mix = nn.Sequential(
            nn.LayerNorm(time_dim),
            FeedForward(time_dim, time_dim // 4, dropout),)
        
        self.freq_mix = nn.Sequential(
            nn.LayerNorm(freq_dim), 
            FeedForward(freq_dim, freq_dim // 2, dropout),)

    def forward(self, x):
        x = x + self.time_mix(x)
        x = x + self.freq_mix(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    

## convolutional mixer block    
class ConvMixerBlock(nn.Module):
    """ Performs convolution with mlp mixer. Processing steps ::
        1) freq depthwise separable convolution
        2) time depthwise separable convolution
        3) mlp mixer
        4) skip connection
    
    """
    def __init__(self, temporal_length, num_temporal_channels=64, 
                 temporal_kernel_size=3, temporal_padding=1,
                 freq_domain_kernel_size=5, freq_domain_padding=2,
                 num_freq_filters=64, 
                 dropout=0.,
                 bias=False):
        super(ConvMixerBlock, self).__init__()

        ## frequency domain encoding
        self.frequency_domain_encoding = nn.Sequential(
            nn.Conv2d(1, num_freq_filters, kernel_size=3, stride=1, padding=1, bias=bias),
            Swish(),
            SeparableConv2d(num_freq_filters, num_freq_filters, 
                            kernel_size=(freq_domain_kernel_size, 1),
                            stride=1, padding=(freq_domain_padding, 0), bias=bias),
            Swish(), 
            nn.Conv2d(num_freq_filters, 1, kernel_size=1, stride=1, padding=0, bias=bias), 
            nn.BatchNorm2d(1),
            Swish(),)
        self.freq_se = SELayer(num_freq_filters, reduction=4, attend_dim="freq_timewise")
        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(num_temporal_channels, num_temporal_channels, 
                            kernel_size=temporal_kernel_size,
                            stride=1, padding=temporal_padding, bias=bias),
            nn.BatchNorm1d(num_temporal_channels),
            Swish(),)
        self.time_se = SELayer(num_temporal_channels, reduction=4, attend_dim="time_freqwise")
        self.dropout = nn.Dropout(p=dropout)
        
        ## mixer
        self.mixer = nn.Sequential(
            MixerBlock(time_dim=temporal_length, freq_dim=num_temporal_channels, dropout=0.),
            Swish(),)
        
    def forward(self, x):
        skipInput = x
        skipInput2 = x = self.dropout(self.frequency_domain_encoding(x.unsqueeze(1)).squeeze(1))
        x = self.dropout(self.temporal_domain_encoding(x))
        x = self.mixer(x)
        
        return skipInput + skipInput2 + x
        
## convolutional mixer block with squeeze and excitation    
class SEConvMixerBlock(nn.Module):
    """ Performs convolution with mlp mixer. Processing steps ::
        1) freq depthwise separable convolution
        2) freq time-wise squeeze and excitation
        3) time depthwise separable convolution
        4) time freq-wise squeeze and excitation
        5) mlp mixer
        6) skip connection
    
    """
    def __init__(self, temporal_length, num_temporal_channels=64, 
                 temporal_kernel_size=3, temporal_padding=1,
                 freq_domain_kernel_size=5, freq_domain_padding=2,
                 num_freq_filters=64, 
                 dropout=0.,
                 bias=False):
        super(SEConvMixerBlock, self).__init__()

        ## frequency domain encoding
        self.frequency_domain_encoding = nn.Sequential(
            nn.Conv2d(1, num_freq_filters, kernel_size=3, stride=1, padding=1, bias=bias),
            Swish(),
            SeparableConv2d(num_freq_filters, num_freq_filters, 
                            kernel_size=(freq_domain_kernel_size, 1),
                            stride=1, padding=(freq_domain_padding, 0), bias=bias),
            Swish(), 
            nn.Conv2d(num_freq_filters, 1, kernel_size=1, stride=1, padding=0, bias=bias), 
            nn.BatchNorm2d(1),
            Swish(),)
        self.freq_se = SELayer(num_freq_filters, reduction=4, attend_dim="freq_timewise")
        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(num_temporal_channels, num_temporal_channels, 
                            kernel_size=temporal_kernel_size,
                            stride=1, padding=temporal_padding, bias=bias),
            nn.BatchNorm1d(num_temporal_channels),
            Swish(),)
        self.time_se = SELayer(num_temporal_channels, reduction=4, attend_dim="time_freqwise")
        self.dropout = nn.Dropout(p=dropout)
        
        ## mixer
        self.mixer = nn.Sequential(
            MixerBlock(time_dim=temporal_length, freq_dim=num_temporal_channels, dropout=0.),
            Swish(),)
        
    def forward(self, x):
        skipInput = x
        skipInput2 = x = self.dropout(self.freq_se(self.frequency_domain_encoding(x.unsqueeze(1))).squeeze(1))
        x = self.dropout(self.time_se(self.temporal_domain_encoding(x)))
        x = self.mixer(x)
        
        return skipInput + skipInput2 + x
    
## convolutional mixer block 
class PreConvBlock(nn.Module):

    def __init__(self, time_length, time_channels=64, 
                 kernel_size=3, padding=1, 
                 dropout=0., bias=False):
        
        super(PreConvBlock, self).__init__()
        
        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(time_channels, time_channels, kernel_size=kernel_size,
                            stride=1, padding=padding, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),
            SeparableConv1d(time_channels, time_channels, kernel_size=1,
                            stride=1, padding=0, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        skipInput = x
        x = self.dropout(self.temporal_domain_encoding(x))
        
        return skipInput + x
class KWSConvMixer(nn.Module):
    def __init__(self, input_size, 
                 num_classes,
                 feat_dim=64,
                 dropout=0.):
        
        """ KWS Convolutional Mixer Model
        input:: audio spectrogram, default input shape [BS, 101, 40]
        output:: prediction of command classes, default 12 classes
        """
        
        super(KWSConvMixer, self).__init__()
        
        self.num_classes = num_classes
        self.temporal_dim, self.frequency_dim = input_size
        
        ## init conv (channel): output shape BS x feat_dim x T        
        self.conv1 = nn.Sequential(
            SeparableConv1d(self.frequency_dim, feat_dim, 
                            kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(feat_dim),
            Swish(),)
        
        self.preConvMixer = PreConvBlock(self.temporal_dim, feat_dim,
                                         kernel_size=7, padding=3,
                                         dropout=dropout)

        self.convMixer1 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=9, temporal_padding=4,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=64, 
                                         dropout=dropout)
        
        self.convMixer2 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=11, temporal_padding=5,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=32, 
                                         dropout=dropout)
            
        self.convMixer3 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=13, temporal_padding=6,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=16, 
                                         dropout=dropout)
        
        self.convMixer4 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=15, temporal_padding=7,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=8, 
                                         dropout=dropout)
        
        self.conv2 = nn.Sequential(
            SeparableConv1d(feat_dim, feat_dim*2, 
                            kernel_size=17, stride=1, padding=8, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv3 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=19, stride=1, padding=18, dilation=2, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv4 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.pooling = torch.nn.AdaptiveMaxPool1d(1) 
        self.mlp_head = nn.Sequential(nn.Linear(feat_dim*2, self.num_classes, bias=True))
        
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.preConvMixer(x)
        x = self.convMixer1(x)
        x = self.convMixer2(x) 
        x = self.convMixer3(x) 
        x = self.convMixer4(x) 

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch, in_channels, timesteps = x.size()
        x = self.pooling(x).view(batch, in_channels)
        return self.mlp_head(x)

class KWSConvMixer_SE(nn.Module):
    def __init__(self, input_size, 
                 num_classes,
                 feat_dim=64,
                 dropout=0.):
        
        """ KWS Convolutional Mixer Model with Squeeze and Excitation
        input:: audio spectrogram, default input shape [BS, 101, 40]
        output:: prediction of command classes, default 12 classes
        """
        
        super(KWSConvMixer_SE, self).__init__()
        
        self.num_classes = num_classes
        self.temporal_dim, self.frequency_dim = input_size
        
        ## init conv (channel): output shape BS x feat_dim x T        
        self.conv1 = nn.Sequential(
            SeparableConv1d(self.frequency_dim, feat_dim, 
                            kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(feat_dim),
            Swish(),)
        self.se1 = SELayer(feat_dim, reduction=4, attend_dim="chan")
        self.preConvMixer = PreConvBlock(self.temporal_dim, feat_dim,
                                         kernel_size=7, padding=3,
                                         dropout=dropout)

        self.convMixer1 = SEConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=9, temporal_padding=4,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=64, 
                                         dropout=dropout)
        
        self.convMixer2 = SEConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=11, temporal_padding=5,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=32, 
                                         dropout=dropout)
            
        self.convMixer3 = SEConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=13, temporal_padding=6,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=16, 
                                         dropout=dropout)
        
        self.convMixer4 = SEConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=15, temporal_padding=7,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=8, 
                                         dropout=dropout)
        
        self.conv2 = nn.Sequential(
            SeparableConv1d(feat_dim, feat_dim*2, 
                            kernel_size=17, stride=1, padding=8, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv3 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=19, stride=1, padding=18, dilation=2, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv4 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.pooling = torch.nn.AdaptiveMaxPool1d(1) 
        self.mlp_head = nn.Sequential(nn.Linear(feat_dim*2, self.num_classes, bias=True))
        
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.preConvMixer(x)
        x = self.convMixer1(x)
        x = self.convMixer2(x) 
        x = self.convMixer3(x) 
        x = self.convMixer4(x) 

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch, in_channels, timesteps = x.size()
        x = self.pooling(x).view(batch, in_channels)
        return self.mlp_head(x)    
    
if __name__=='__main__':

    model = KWSConvMixer_SE(input_size = (98, 64), 
                         num_classes=12,
                         feat_dim=64,
                         dropout=0.0)