
import os
import time
import torch
from utils import *
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from cnn import CNN
from model2 import My_Model

import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # acc1 = accuracy(tmp_pred, tmp_label)
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


class Trainer(object):
    def __init__(self, data_loader, test_data, config):

        # Data loader
        self.data_loader = data_loader
        self.test_data = test_data

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version


        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()



        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def test(self):
        # data_iter = iter(self.test_data)
        # try:
        #     images, z = next(data_iter)
        # except:
        #     data_iter = iter(self.test_data)
        #     images, z = next(data_iter)
        total = 0
        correct = 0
        for step, (images, z) in enumerate(self.test_data):
            images = tensor2var(images)
            outputs = self.resnet(images)
            predicts = np.empty((images.shape[0], 18))
            for i in range(18):
                predicts[:, i] = var2numpy(torch.max(outputs[i].data, 1)[1]).squeeze()
                # if i == 0:
                #     predicts = var2numpy(torch.max(outputs[i].data, 1)[1]).squeeze()
                #     predicts_temp = predicts
                # else:
                #     temp = var2numpy(torch.max(outputs[i].data, 1)[1]).squeeze()
                #     predicts = np.concatenate((predicts_temp, temp), axis=1)
                #     predicts_temp = predicts
            total += z.shape[0]
            for i in range(images.shape[0]):
                if (predicts[i] == var2numpy(z)[i]).all() == True:
                    correct = correct + 1
            a = 0
        print(total)
        print(correct)
        print(100 * correct / total)



        #         total += z.size(0)
        #         correct += (predicts == y.numpy()).sum()
        #     # print('Accuracy = %.2f' % (100 * correct / total))
        #
        # return 100 * correct / total

    def train(self):
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)


        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            # self.cnn.train()
            self.resnet.train()

            try:
                real_images, z = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, z = next(data_iter)
                a = 0

            real_images = tensor2var(real_images)
            z = tensor2var(z)
            # output = self.cnn(real_images)
            output = self.resnet(real_images)
            loss_total = 0
            for i in range(18):
                # loss_total = loss_total + self.cnn_loss_function(output[i], z[:, i])
                loss_total = loss_total + self.resnet_loss_function(output[i], z[:, i])
            self.reset_grad()
            loss_total.backward()
            # self.cnn_optimizer.step()
            self.resnet_optimizer.step()

            # if step % 200 == 0:
            #     torch.save(self.cnn.state_dict(),
            #                os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))

            # if (step+1) % 2 == 0:
            #     print('******************testing******************')
            #     self.test()
            predicts = np.empty((real_images.shape[0], 18))
            for i in range(18):
                predicts[:, i] = var2numpy(torch.max(output[i].data, 1)[1]).squeeze()
            if step % 20 == 0:
                total_acc = 0
                for i in range(18):
                    tmp_pred = output[i]
                    tmp_label = z[:, i]
                    acc0 = accuracy(tmp_pred, tmp_label)
                    total_acc = total_acc + var2numpy(acc0[0].data)
                print('avg_acc=')
                print(step)
                print(float(total_acc/18))


            # correct = 0
            # for i in range(real_images.shape[0]):
                # if (predicts[i] == var2numpy(z)[i]).all() == True:
                #     correct = correct + 1
            # a = 0
            # print(real_images.shape[0])
            # print(correct)
            # print(100 * correct / real_images.shape[0])



    def build_model(self):
        # self.cnn = CNN().cuda()
        # self.cnn_optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.001)
        # self.cnn_loss_function = torch.nn.CrossEntropyLoss()

        self.resnet = My_Model().cuda()
        self.resnet_optimizer = torch.optim.Adam(self.resnet.parameters(), lr=0.001)
        self.resnet_loss_function = torch.nn.CrossEntropyLoss()

        print(self.resnet)
        a = 0



    def load_pretrained_model(self):
        self.cnn.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        # self.cnn_optimizer.zero_grad()
        self.resnet_optimizer.zero_grad()


    # def save_sample(self, data_iter):
    #     real_images, _ = next(data_iter)
    #     save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
