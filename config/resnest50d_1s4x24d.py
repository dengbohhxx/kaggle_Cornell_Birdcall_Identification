# -*- coding: utf-8 -*-
import torch
model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 512,
    "fmin": 20,
    "fmax": 16000,
    "classes_num": 264
}
class TrainGlobalConfig:
    device='cpu'
    k_fold=5
    num_workers = 2
    batch_size = 16
    n_epochs = 40  # n_epochs = 40
    lr =0.0005
    folder = 'output'
    verbose = True
    verbose_step = 1
    #label_smoothing
    label_smoothing=True
    eps=0.05  
    validation_scheduler = True  # do scheduler.step after validation stage loss
    step_scheduler = False  # do scheduler.step after optimizer.step
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08      
        )