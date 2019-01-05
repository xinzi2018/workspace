
#coding: utf-8 
'''train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

# 获取参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')#表示继续训练模型
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 获取数据集，并先进行预处理
print('==> Preparing data..')
# 图像预处理和增强
transform_train = transforms.Compose([  #compose表示将以下几种变换组合起来
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),#以给定的概率随机水平翻转图片
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#用平均值和标准偏差归一化张量图像
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
							
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 继续训练模型或新建一个模型
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = VGG('VGG16')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

# 如果GPU可用，使用GPU
if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    # parallel use GPU
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()-1))
    # speed up slightly
    cudnn.benchmark = True


# 定义度量和优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 训练阶段
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0# 用来统计数据集的个数
    # batch 数据
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 将数据移到GPU上
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 先将optimizer梯度先置为0
        optimizer.zero_grad()
        # Variable表示该变量属于计算图的一部分，此处是图计算的开始处。图的leaf variable
        inputs, targets = Variable(inputs), Variable(targets)
        # 模型输出
        outputs = net(inputs)
        # 计算loss，图的终点处
        loss = criterion(outputs, targets)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 注意如果你想统计loss，切勿直接使用loss相加，而是使用loss.data[0]。因为loss是计算图的一部分，如果你直接加loss，代表total loss同样属于模型一部分，那么图就越来越大
        train_loss += loss.data[0]
        # 数据统计
        _, predicted = torch.max(outputs.data, 1)#max的输出是一个元组，第一个数据是每一行最大的这个值，第二个数据是最大数值对应的下标。
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()#correct表示分类正确的个数。

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# 测试阶段
def test(epoch):
    global best_acc  #global 表示为全局变量
    # 先切到测试模型
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # 保存模型
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

# 运行模型
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    # 清除部分无用变量 
    torch.cuda.empty_cache() 

