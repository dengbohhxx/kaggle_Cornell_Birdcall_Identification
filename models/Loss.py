#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Loss Functions
import torch
import torch.nn as nn

class PANNsLoss(nn.Module):
    def __init__(self,label_smoothing=None,epsilon=0.05):
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
        input_ = input["clipwise_output"]

        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()
        print(input_)
        print(target)
        input_argmax = torch.argmax(input_, dim=1)
        target_argmax = torch.argmax(target, dim=1)
        print('input_argmax: ', input_argmax)
        print('target_argmax: ', target_argmax)
        input_max = torch.max(input_, dim=1)[0]
        target_max = torch.max(target, dim=1)[0]
        print('input_max: ', input_max)
        print('target_max: ', target_max)
        if self.label_smoothing!=None:
            target=self.label_smooth(target)

        return self.bce(input_, target), input["clr_out"]

class Contrastive_loss(nn.Module):
    '''
    contrastive loss of simCLR
    '''
    def __init__(self, batch_size, temperature=1.0):
        super(Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=2)
        self.mask = self.mask_correlated()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        '''
        Given a positive pair, treat the other 2(batch_size - 1) augmented example as negative examples
        '''
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        # cosine similarity
        similar = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # NT_Xent loss matrix
        sim_i_j = torch.diag(similar, self.batch_size)
        sim_j_i = torch.diag(similar, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = similar[self.mask].reshape(N, -1)
        logits_matrix = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(N).to(logits_matrix.device).long()
        # loss
        loss = self.criterion(logits_matrix, labels)
        loss /= N

        return loss
