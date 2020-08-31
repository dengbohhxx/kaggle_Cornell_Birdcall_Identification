# -*- coding: utf-8 -*-
import torch
model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 8000,
    "classes_num": 264
}
class TrainGlobalConfig:
    device='cuda:0'
    k_fold=5
    num_workers = 6
    batch_size = 16
    n_epochs = 40  # n_epochs = 40
    lr = 0.005
    folder = 'output'
    verbose = True
    verbose_step = 1
    #label_smoothing
    label_smoothing=False
    eps=0.05  
    
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08      
    )
