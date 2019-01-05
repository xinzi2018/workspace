#coding:utf-8						
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

from torch.autograd import Variable
from model import *

parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0
start_epoch = 0


print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/xinzi/dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/xinzi/dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(trainset)
print(testset)



if args.resume:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	net = checkpoint['net']
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']
else:
	print('==> Building model..')
	net = VGG('VGG16')
	print(net)


use_cuda = torch.cuda.is_available()
if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()-1))
	cudnn.benchmark = True #设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最合适当前的配置的高效算法，来达到优化运行效率的问题。


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)


def train(epochs):
	i = 0
	net.train()
	for epoch in range(epochs):
		print('\nEpoch: %d' % epoch)

	
		for batch_idx, (inputs, targets) in enumerate(trainloader):
			i = i + 1
			if use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda()
			optimizer.zero_grad()
			inputs, targets = Variable(inputs), Variable(targets)
		
			outputs = net(inputs)
		
			loss = criterion(outputs, targets)
			loss.backward()

			optimizer.step()

	print('total i=%d' %i)
	print('end loss=%3f' %loss.data)



def mytest():
	
	net.eval()
	
	total = 0
	test_loss = 0
	correct = 0


	for batch_idx, (inputs, targets) in enumerate(testloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True),Variable(targets)#volatile为ture表示所有依它的节点该属性都为true，表示不会求导。该属性的优先级高于requires_grad

		outputs=net(inputs)

		loss = criterion(outputs, targets)
		test_loss += loss.data[0]#注意该写法
		_, predicted = torch.max(outputs.data, 1)#注意outputs数据的提取
		total += targets.size(0)#注意targets大小的表达形式
		correct += predicted.eq(targets.data).cpu().sum()

	print('result')
	#print(total)
	print(correct)


if __name__ == '__main__':
	i = 1
	#for epoch in range(20):
	#	print('i=%d' %i)
	#	train(i)
	#	test()
	#	i = i+5
	train(i)
	mytest()






 










