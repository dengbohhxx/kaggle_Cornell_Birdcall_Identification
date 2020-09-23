#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Loss Functions
import torch
import torch.nn as nn
import torch.nn.functional as F
class sig_bce_Loss(nn.Module):
    def __init__(self,label_smoothing=False,epsilon=0.05,alpha=0.99):
        super().__init__()

        self.bce = nn.BCELoss()
        self.label_smoothing=label_smoothing
        self.epsilon=epsilon
        self.alpha = alpha
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
        target = target.float()
        if self.label_smoothing!=False:
            target=self.label_smooth(target)
        loss = F.binary_cross_entropy_with_logits(input_, target, reduction='none')
        loss = loss.mean(dim=1)
        with torch.no_grad():
            outlier_mask = loss > self.alpha * loss.max()
            if (outlier_mask +0).sum().detach().item()==list(loss.size())[0]:
                all_t=True
            else:
                all_t=False
            outlier_idx = (outlier_mask == 0).nonzero().squeeze(1)
        if all_t==True:
            loss=loss.mean()
            return loss
        loss = loss[outlier_idx].mean()
        return loss
