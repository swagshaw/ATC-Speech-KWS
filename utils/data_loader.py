"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午4:59
@Author  : Yang "Jan" Xiao 
@Description : data_loader
"""
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_audiomentations import Compose, AddColoredNoise
class SpeechCommandDataset(Dataset):
    def __init__(self, root, filename, is_training, class_list, class_encoding, noise_augmentation=None):
        super(SpeechCommandDataset, self).__init__()
        """
        Args:
            root: "./data root"
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.classes = class_list
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.root = root
        self.filename = filename
        self.is_training = is_training
        self.class_encoding = class_encoding
        self.speech_dataset = self.combined_path()
        self.noise_augmentation = noise_augmentation
        if self.noise_augmentation is not None:
            print("Noise augmentation is enabled")
            print(f"SNR range: {noise_augmentation} - {noise_augmentation + 5} dB")
            self.augment = AddColoredNoise(min_snr_in_db=noise_augmentation, max_snr_in_db=noise_augmentation + 5, p=1.0).cuda()
    def combined_path(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/")
            if "unknown" in self.classes and wave_name == "silence":
                dataset_list.append(["silence", "unknown"])
            elif category in self.classes:
                path = os.path.join(self.root, category, wave_name)
                dataset_list.append([path, category])
        return dataset_list

    def load_audio(self, speech_path):
        waveform, sr = torchaudio.load(speech_path)
        if waveform.shape[1] < self.sample_length:
            # padding if the audio length is smaller than samping length.
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
        else:
            waveform = waveform.narrow(1, 0, self.sample_length)
        if self.noise_augmentation is not None:
                waveform = self.augment(waveform.unsqueeze(0), sample_rate = sr).squeeze(0)
        if self.is_training:
            pad_length = int(waveform.shape[1] * 0.1)
            waveform = F.pad(waveform, [pad_length, pad_length])
            offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
            waveform = waveform.narrow(1, offset, self.sample_length)
        return waveform

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_path = self.speech_dataset[index][0]
        speech_category = self.speech_dataset[index][1]
        label = self.one_hot(speech_category)

        if speech_path == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label
