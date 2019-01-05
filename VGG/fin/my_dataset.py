import torch
import os
import numpy as np
from PIL import Image
import torch.utils.data as data


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_dataset(root_path):
    all_imgs = os.listdir(root_path)
    imgs = []
    labels = []
    for tmp_img in all_imgs:
        all_path = os.path.join(root_path, tmp_img)
        if tmp_img.split(".")[0] == 'cat':
            labels.append(0)
        else:
            labels.append(1)
        imgs.append(all_path)
    return imgs, labels


class MyDataset(data.Dataset):
    def __init__(self, root_path, img_transform=None, img_loader=pil_loader):
        self.path = root_path
        self.transform = img_transform
        self.loder = img_loader
        self.data, self.labels = make_dataset(self.path)

    def __getitem__(self, index):
        path = self.data[index]
        label = self.labels[index]
        img = self.loder(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
