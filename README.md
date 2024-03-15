# ATC-Speech-KWS
AI Research into Spoken Keyword Spotting. 
Collection of PyTorch implementations of Spoken Keyword Spotting presented in research papers.
Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. 

# 目录
  * [配置](#installation)
  * [模型列表](#implementations)
    + [Temporal Convolution Resnet(TC-ResNet)](#temporal-convolution-resnet)
    + [Broadcasting Residual Network(BC-ResNet)](#broadcasting-residual-network)
    + [MatchboxNet](#matchboxnet)
    + [ConvMixer](#convmixer)
    + [KWT](#kwt)


# Installation

```bash
首先，把数据集的data文件夹放到：<当前文件夹>/dataset/atc/data
```


# Implementations
一共有五个基线模型，包括了各自的论文地址
## Temporal Convolution Resnet
_Temporal Convolution for Real-time Keyword Spotting on Mobile Devices_
[[Paper]](https://arxiv.org/abs/1904.03814) [[Code]](networks/tcresnet.py)

## Broadcasting Residual Network
_Broadcasted Residual Learning for Efficient Keyword Spotting_
[[Paper]](https://arxiv.org/abs/2106.04140) [[Code]](networks/bcresnet.py)

## MatchboxNet
_MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition_
[[Paper]](https://arxiv.org/abs/2004.08531) [[Code]](networks/matchboxnet.py)

## ConvMixer
_ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-field Keyword Spotting_
[[Paper]](https://arxiv.org/abs/2201.05863) [[Code]](networks/convmixer.py)

## KWT
_Keyword transformer: A self-attention model for keyword spotting_
[[Paper]](https://arxiv.org/abs/2104.00769) [[Code]](network/kwt.py)

# Experiments
## 实验一 模型性能比较
```bash
bash e1.sh
```
## 实验二 不同噪声情况下的训练
```bash
bash e2.sh
```
## 实验三 不同数据集大小情况下的训练
```bash
bash e3.sh
```
