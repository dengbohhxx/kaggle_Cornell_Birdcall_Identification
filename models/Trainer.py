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
        self.mixup = True
        self.pre_x = torch.empty((30, 160000))
        self.pre_y = torch.empty((30, 264))
    def forward(self,x,label):
        print(x.shape)
        print(label.shape)
        if self.mixup:
            x_mixup = do_mixup(x.cpu(), self.pre_x.cpu(), mixup_lambda=0.3).cuda()
            y_mixup = do_mixup(label.cpu(), self.pre_y.cpu(), mixup_lambda=0.3).cuda()
            print(x_mixup.shape)
            print(y_mixup.shape)
            print(x)
            print(label)
            print(x_mixup)
            print(y_mixup)
            self.pre_x = x
            self.pre_y = label
            predicts = self.model(x_mixup)
            return self.loss_F(predicts,y_mixup)
        else:
            predicts = self.model(x)
            return self.loss_F(predicts,label)