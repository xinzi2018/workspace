import pretrainedmodels
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        model_name = 'vgg11_bn'
        self.base_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.ls = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.ls(x)
        return x
