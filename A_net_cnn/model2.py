# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class My_Model(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''

    def __init__(self, num_classes=1000):
        super(My_Model, self).__init__()

        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

        self.linear_to = nn.Linear(512 * 7 * 7, 2048)
        self.linear_to2 = nn.Linear(2048, 1024)

        linear_n = []
        classes = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
        for i in classes:
            linear_n += [nn.Linear(1024, i)]
        self.linears = nn.Sequential(*linear_n)

    def _make_layer(self, inchannel, outchannel, bloch_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, bloch_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.linear_to(x)
        x = self.linear_to2(x)
        for i in range(18):
            output.append(self.linears._modules[str(i)](x))
        return output


model = My_Model()
print(model)
