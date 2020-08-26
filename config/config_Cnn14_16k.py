# -*- coding: utf-8 -*-
import torch
model_config = {
    "sample_rate": 16000,
    "window_size": 512,
    "hop_size": 160,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 8000,
    "classes_num": 264
}
class TrainGlobalConfig:
    num_workers = 2
    batch_size = 2
    n_epochs = 40  # n_epochs = 40
    lr = 0.0004

    folder = 'log'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    # 我们只在每次 epoch 完，验证完后，再执行学习策略。
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    # 当指标变化小时，减少学习率
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