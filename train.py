#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler,SequentialSampler

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from dataset.mp3_dataset import Ori_Mp3_Dataset
from dataset.wav_dataset import Ori_Wav_Dataset
from utlis.fitter import Fitter
from models.backbones import  Cnn14_16k, ResNet38, Wavegram_Cnn14, Wavegram_Logmel_Cnn14
from config.config_ResNet38_relu_softmax_mixup import model_config, TrainGlobalConfig
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
RAW_DATA = ROOT / "birdsong-recognition/resample_dataset_32k"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio_resampled"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###k折交叉验证###
train_data = pd.read_csv(RAW_DATA/"train.csv")
tmp_list = []
for ebird_d in TRAIN_AUDIO_DIR.iterdir():
    for audio_file in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, audio_file.name, audio_file.as_posix()])
train_audio_path_exist = pd.DataFrame(tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])
del tmp_list
train_all = pd.merge(
    train_data, train_audio_path_exist, on=["ebird_code", "resampled_filename"], how="inner")
train_all.loc[:, 'fold'] = -1
skf = StratifiedKFold(n_splits=TrainGlobalConfig.k_fold, shuffle=True, random_state=1024)
for fold_number, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.loc[train_all.iloc[val_index].index, 'fold'] = fold_number
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###构建模型###
model=ResNet38(**model_config)
checkpoint=torch.load("./pretrained_weight/ResNet38_mAP=0.434.pth",map_location="cpu")
pretrained_weights=checkpoint["model"]
model_dict=model.state_dict()
new_dict = {k: v for k, v in pretrained_weights.items() if k.find("fc_audioset")==-1 and k in model_dict.keys()}
model_dict.update(new_dict)
model.load_state_dict(model_dict)
net=trainer(model,PANNsLoss(TrainGlobalConfig.label_smoothing,TrainGlobalConfig.eps))   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def collate_fn(batch):
    return tuple(zip(*batch))
device = torch.device(TrainGlobalConfig.device)
net.to(device)
fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
for i in range(TrainGlobalConfig.n_epochs):    
    fold_number = i%TrainGlobalConfig.k_fold
    train_dataset = Ori_Wav_Dataset(
        sounds_id=train_all[train_all['fold'] != fold_number].index.values,
        train_all=train_all,
        waveform_transforms=None,
    )
    validation_dataset = Ori_Wav_Dataset(
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

