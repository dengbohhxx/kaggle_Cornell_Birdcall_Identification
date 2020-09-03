#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
"""
AdaBoost algorithm for simple ensemble learning
"""

class AdaBoost_simple():
    '''
    models: ensemble models
    dataloader: provide tensor data
    '''
    def __init__(self, models, dataloader):
        self.models = models
        self.dataloader = dataloader
        self.ori_weight = 1.0 / len(dataloader)
        self.error = dict
        self.alpha = dict

    def each_error(self, data):
        inputs = data[0]
        labels = data[1]
        for model in self.models:
            predicts = model(inputs)
            is_error = predicts != labels
            self.error[model.__name__] += np.sum(predicts[is_error].numpy()) * self.ori_weight #TODO: remove predict value,just use bool

    def calculate_coefficient(self):
        for step, (waveform, labels) in enumerate(self.dataloader):
            data = (waveform, labels)
            self.each_error(data)
        for name, error in self.error.items():
            self.alpha[name] = 0.5 * np.log((1 - error) / (error))

    def __call__(self):
        self.calculate_coefficient()
        return self.alpha

