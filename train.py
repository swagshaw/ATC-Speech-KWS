"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午4:57
@Author  : Yang "Jan" Xiao 
@Description : train
"""
import logging
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import *
from utils.utils import *
from torch.utils.data import DataLoader, Subset, random_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain
logger = logging.getLogger()


def get_dataloader_keyword(data_path, class_list, class_encoding, parameters):
    """
    CL task protocol: keyword split.
    To get the GSC data and build the data loader from a list of keywords.
    """
    batch_size = parameters.batch
    trainratio = parameters.trainratio
    if len(class_list) != 0:
        if parameters.unknown_ratio > 0.0:
            train_filename = readlines(f"{data_path}/train_withunkeyword.txt")
            test_filename = readlines(f"{data_path}/test_withunkeyword.txt")
        else:
            train_filename = readlines(f"{data_path}/train.txt")
            test_filename = readlines(f"{data_path}/test.txt")
        train_dataset = SpeechCommandDataset(f"{data_path}/data", train_filename, True, class_list, class_encoding, parameters.noise)
        test_dataset = SpeechCommandDataset(f"{data_path}/data", test_filename, False, class_list, class_encoding, parameters.noise)
               # Split train_dataset into training and validation sets
        num_train = len(train_dataset)
        num_valid = int(num_train * 0.2)
        train_subset, valid_subset = random_split(train_dataset, [num_train - num_valid, num_valid])
        train_subset = Subset(train_dataset, range(int(trainratio * len(train_subset))))

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_dataloader, valid_dataloader, test_dataloader
    else:
        raise ValueError("the class list is empty!")


class Trainer:
    """
    The KWS model training class.
    """

    def __init__(self, opt, model):
        self.opt = opt
        self.lr = opt.lr
        self.step = opt.step
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.model = model
        self.device, self.device_list = prepare_device(opt.gpu)
        self.best_acc = 0.0
        # map the model weight to the device.
        self.model.to(self.device)
        # enable multi GPU training.
        if len(self.device_list) > 1:
            print(f">>>   Available GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = {
            "train_loss": 0.0, "train_accuracy": 0.0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0.0, "valid_accuracy": 0.0, "valid_total": 0, "valid_correct": 0}
     
    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # if (self.epo + 1) % self.opt.freq == 0:
        #     torch.save(self.model.state_dict(), os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))
        # save the best model.
        if self.loss_name["valid_accuracy"] > self.best_acc:
            self.opt.best_acc = self.loss_name["valid_accuracy"]
            torch.save(self.model.state_dict(), os.path.join(save_directory, f"best_in_{self.epoch}.pt"))
        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))

    def model_train(self, optimizer, scheduler, train_dataloader, valid_dataloader):
        """
        Normal model training process, without modifying the loss function.
        """
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        logger.info(f"[3] Training for {self.epoch} epochs...")
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.cuda(self.device)
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader), position=0, total=len(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                """
                waveform:(B,C,16000)
                MFCC:(B,C,F,T)
                labels:(B,)
                """
                # print((waveform.size,labels))
                optimizer.zero_grad()
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in enumerate(valid_dataloader):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
            scheduler.step()
            self.model_save()
            logger.info(
                f"Epoch {self.epo + 1}/{self.epoch} | train_loss {self.loss_name['train_loss']:.4f} "
                f"| train_acc {100 * self.loss_name['train_accuracy']:.4f} | "
                f"valid_loss {self.loss_name['valid_loss']:.4f} "
                f"| valid_accuracy {100 * self.loss_name['valid_accuracy']:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.4f}"
            )
        return self.loss_name

    def model_test(self, test_dataloader):
        self.model.eval()
        test_length = len(test_dataloader)
        self.loss_name.update({key: 0 for key in self.loss_name})
        for batch_idx, (waveform, labels) in enumerate(test_dataloader):
            with torch.no_grad():
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)

                self.loss_name["valid_loss"] += loss.item() / test_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["valid_total"] += labels.size(0)
                self.loss_name["valid_correct"] += (predict == labels).sum().item()
                self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
                self.loss_name["f1_score"] = f1_score(labels.cpu().numpy(), predict.cpu().numpy(), average='macro')
        logger.info(
            f"test_loss {self.loss_name['valid_loss']:.4f} "
            f"| test_acc {self.loss_name['valid_accuracy']:.4f}"
            f"| f1_score {self.loss_name['f1_score']:.4f}"
        )
        return self.loss_name