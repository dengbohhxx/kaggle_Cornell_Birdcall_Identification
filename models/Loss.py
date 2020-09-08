#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Loss Functions
import torch
import torch.nn as nn
class sig_bce_Loss(nn.Module):
    def __init__(self,label_smoothing=False,epsilon=0.05):
        super().__init__()

        self.bce = nn.BCELoss()
        self.label_smoothing=label_smoothing
        self.epsilon=epsilon
    def label_smooth(self,label):
        label=label+self.epsilon/label.size()[-1]
        _,index=torch.max(label,-1)  
        for i in range(label.size()[0]):
            label[i][index[i]]=1-self.epsilon
        return label    
    def forward(self, input, target):
        input_ = input
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = nn.functional.sigmoid(input)
        target = target.float()
        if self.label_smoothing!=False:
            target=self.label_smooth(target)
        return self.bce(input_, target)
