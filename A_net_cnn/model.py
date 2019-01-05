# -*- coding: utf-8 -*-
# File  : model.py
# Author: Zhenying Fang
# Date  : 18-12-5

import pretrainedmodels
import torch
import torch.nn as nn


class My_Model(nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()

        self.base_model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')

        self.linear_to = nn.Linear(512*7*7, 2048)
        self.linear_to2 = nn.Linear(2048, 1024)

        linear_n = []
        classes = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
        for i in classes:
            linear_n += [nn.Linear(1024, i)]
        self.linears = nn.Sequential(*linear_n)

    def forward(self, input):
        output = []
        base_features = self.base_model.features(input)
        base_features = base_features.view(base_features.size(0), -1)
        base_features = self.linear_to(base_features)
        base_features = self.linear_to2(base_features)
        for i in range(18):
            output.append(self.linears._modules[str(i)](base_features))
        return output

model = My_Model()
print(model)