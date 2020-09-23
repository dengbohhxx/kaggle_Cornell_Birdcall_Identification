#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
import random
import timm
from timm import create_model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler,SequentialSampler
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from dataset.mp3_dataset import Mp3_LogmelDataset
from utlis.fitter import Fitter
from config.efficientnet_b5 import model_config,TrainGlobalConfig
from models.Trainer import trainer
from models.Loss import sig_bce_Loss
from torchsummary import summary
from timm.models.layers.classifier import create_classifier
from dataset.mp3_dataset import RandomMixer, AddMixer, SigmoidConcatMixer, UseMixerWithProb
from dataset.transforms import get_transforms
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###设定种子###
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore       
set_seed(1024)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###路径设置###
ROOT = Path.cwd()
print(ROOT)
RAW_DATA = ROOT / "birdsong-recognition"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###k折交叉验证###
train_data = pd.read_csv(RAW_DATA/"train.csv")
tmp_list = []
for ebird_d in TRAIN_AUDIO_DIR.iterdir():
    for audio_file in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, audio_file.name, audio_file.as_posix()])
train_audio_path_exist = pd.DataFrame(tmp_list, columns=["ebird_code", "filename", "file_path"])
del tmp_list
train_all = pd.merge(
    train_data, train_audio_path_exist, on=["ebird_code", "filename"], how="inner")
train_all.loc[:, 'fold'] = -1
skf = StratifiedKFold(n_splits=TrainGlobalConfig.k_fold, shuffle=True, random_state=1024)
for fold_number, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.loc[train_all.iloc[val_index].index, 'fold'] = fold_number
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###构建模型###
model=create_model(model_name="tf_efficientnet_b5_ns")
checkpoint=torch.load('./pretrained_weight/tf_efficientnet_b5_ns-6f26d0cf.pth',map_location="cpu")
model.load_state_dict(checkpoint)
'''
model.classifier=nn.Sequential(
            nn.Conv2d(2048, 1024,1), nn.ReLU(),
            nn.Conv2d(1024, 1024,1), nn.ReLU(),
            nn.Conv2d(1024, model_config["classes_num"],1))
'''
model.global_pool,model.classifier=create_classifier(num_features=2048,
                                                     num_classes=model_config["classes_num"]
                                                     , pool_type='avg', use_conv=True)
net=trainer(model,sig_bce_Loss(TrainGlobalConfig.label_smoothing,TrainGlobalConfig.eps))   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def collate_fn(batch):
    return tuple(zip(*batch))
device = torch.device(TrainGlobalConfig.device)
net.to(device)
fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
train_transfrom = get_transforms(train=True,
                                     size=256,
                                     wrap_pad_prob=0.5,
                                     resize_scale=(0.8, 1.0),
                                     resize_ratio=(1.7, 2.3),
                                     resize_prob=0.33,
                                     spec_num_mask=2,
                                     spec_freq_masking=0.15,
                                     spec_time_masking=0.20,
                                     spec_prob=0.5)
mixer = RandomMixer([
        SigmoidConcatMixer(sigmoid_range=(3, 12)),
        AddMixer(alpha_dist='uniform')
    ], p=[0.6, 0.4])
mixer = UseMixerWithProb(mixer, prob=0.8)
for i in range(TrainGlobalConfig.n_epochs):    
    fold_number = 0
    train_dataset = Mp3_LogmelDataset(
        sounds_id=train_all[train_all['fold'] != fold_number].index.values,
        train_all=train_all,
        waveform_transforms=None,
        mixer=mixer,
        spectrogram_transforms=train_transfrom
    )
    validation_dataset = Mp3_LogmelDataset(
        sounds_id=train_all[train_all['fold'] == fold_number].index.values,
        train_all=train_all,
        waveform_transforms=None,
        spectrogram_transforms=get_transforms(False, 256)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    fitter.fit(train_loader, val_loader)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

