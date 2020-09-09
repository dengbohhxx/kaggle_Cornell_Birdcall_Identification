#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###Trainer.py

from utlis.pytorch_utils import do_mixup
import torch

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
class trainer(nn.Module):
    def __init__(self,model,loss_F):
        super().__init__()
        self.model=model
        self.loss_F=loss_F
    def forward(self,x,label):
        predicts = self.model(x)
        return self.loss_F(predicts,label)
