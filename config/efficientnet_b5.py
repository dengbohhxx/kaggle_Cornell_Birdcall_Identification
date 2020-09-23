# -*- coding: utf-8 -*-
import torch
model_config = {
    "sample_rate": 32000,
    "duration":5,
    "window_size": 2560,
    "hop_size":690 ,
    "mel_bins": 128,
    "fmin": 20,
    "fmax": 16000,
    "classes_num": 264
    
}
class TrainGlobalConfig:
    device='cpu'
    k_fold=5
    num_workers = 2
    batch_size = 16
    n_epochs = 50  # n_epochs = 40
    lr =0.0006
    folder = 'output'
    verbose = True
    verbose_step = 1
    #label_smoothing
    label_smoothing=True
    eps=0.05  
    validation_scheduler = True  # do scheduler.step after validation stage loss
    step_scheduler = False  # do scheduler.step after optimizer.step
    warm_up_epochs=10