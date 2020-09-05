# -*- coding: utf-8 -*-
model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 264
}
class TrainGlobalConfig:
    device='cpu'
    k_fold=5
    num_workers = 2
    batch_size = 16
    n_epochs = 40  # n_epochs = 40
    lr =0.0004
    folder = 'output'
    verbose = True
    verbose_step = 1
    #label_smoothing
    label_smoothing=True
    eps=0.05  
    validation_scheduler = True  # do scheduler.step after validation stage loss
