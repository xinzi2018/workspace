# -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable
import numpy as np
from my_dataset import MyDataset
from mymodel import MyModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch.optim as optim
import torch.nn as nn
#from tensorboardX import SummaryWriter
import torch.nn.functional as F
import shutil
import torch.backends.cudnn as cudnn
import argparse
import time
from PIL.Image import BILINEAR

#writer = SummaryWriter("log")

parser = argparse.ArgumentParser(description="demo for pytorch classification")
parser.add_argument('--train_path', type=str, default="data/train")
parser.add_argument('--val_path', type=str, default="data/validation")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--workers', type=float, default=4)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--print-freq', '-p', default=20, type=int)
parser.add_argument('--eval_freq', type=int, default=1)

args = parser.parse_args()


def main():
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = MyModel()
    model.cuda()
    cudnn.benchmark = True
    '''

    train_loader = torch.utils.data.DataLoader(
        MyDataset(args.train_path, img_transform=Compose([
            Resize(256, interpolation=BILINEAR),
            CenterCrop(224),
            ToTensor(),
            Normalize(
                mean=[.485, .456, .406],
                std=[.229, .224, .225]
            )
        ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        a=0
    '''
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': model.base_model.parameters(), 'lr': args.lr * 0.1},
                           {'params': model.ls.parameters()}],
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
                          )

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            # is_best = prec1 > best_prec1
            # best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_prec1,
            }, False, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0].cpu().numpy()[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr'])))
    writer.add_scalar("Train/Acc", top1.avg, epoch)
    writer.add_scalar("Train/Loss", losses.avg, epoch)


def adjust_learning_rate(optimizer, epoch):
    if epoch == 50 or epoch == 55:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    filename = '_'.join((str(epoch), filename))
    filename = "trained_model/" + filename
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    '''

if __name__ == '__main__':
    main()
