#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###Trainer.py
"""
class Trainer(nn.module):
    def __init__(self,models,loss,label):
        self.model=models
        self.loss=loss
        self.labels=label
    def forward(voice,label): 
        predict=self.model(voice)
        loss=self.loss(predict,self.labels)
        return loss           
"""
import torch.nn as nn
from config.resnet54 import TrainGlobalConfig
class trainer(nn.Module):
    def __init__(self,model,loss_F):
        super().__init__()
        self.model=model
        self.loss_F=loss_F
    def forward(self,x,label):
        inputs=self.model(x,TrainGlobalConfig.batch_size)
        return self.loss_F(inputs,label)