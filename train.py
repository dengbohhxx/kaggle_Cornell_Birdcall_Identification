#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import cv2
import audioread

import os
import random

import librosa
import librosa.display as display
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler,SequentialSampler

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from dataset.mp3_dataset import Ori_Mp3_Dataset
from utlis.fitter import Fitter
from models.backbones import  Cnn14_16k 
from config.config_Cnn14_16k import model_config,TrainGlobalConfig
from models.Trainer import trainer
from models.Loss import PANNsLoss
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
    for mp3_file in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, mp3_file.name, mp3_file.as_posix()])
train_mp3_path_exist = pd.DataFrame(tmp_list, columns=["ebird_code", "filename", "file_path"])
del tmp_list
train_all = pd.merge(
    train_data, train_mp3_path_exist, on=["ebird_code", "filename"], how="inner")
train_all.loc[:, 'fold'] = 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
for fold_number, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.loc[train_all.iloc[val_index].index, 'fold'] = fold_number
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###构建模型###
model=Cnn14_16k(**model_config)
checkpoint=torch.load("./pretrained_weight/Cnn14_16k_mAP=0.438.pth",map_location="cpu")
pretrained_weights=checkpoint["model"]
model_dict=model.state_dict()
new_dict = {k: v for k, v in pretrained_weights.items() if k.find("fc_audioset")==-1 and k in model_dict.keys()}
model_dict.update(new_dict)
model.load_state_dict(model_dict)
net=trainer(model,PANNsLoss())   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def collate_fn(batch):
    return tuple(zip(*batch))
device = torch.device(TrainGlobalConfig.device)
net.to(device)
fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
for i in range(TrainGlobalConfig.n_epochs):    
    fold_number = i%5
    train_dataset = Ori_Mp3_Dataset(
        sounds_id=train_all[train_all['fold'] != fold_number].index.values,
        train_all=train_all,
        waveform_transforms=None,
    )

    validation_dataset = Ori_Mp3_Dataset(
        sounds_id=train_all[train_all['fold'] == fold_number].index.values,
        train_all=train_all,
        waveform_transforms=None,
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

