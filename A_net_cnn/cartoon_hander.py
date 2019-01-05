# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
import csv
import torch
import torch.utils.data as data

'''
def png2jpg(path):
    a = os.listdir(path)
    a.sort()
    for file in a:
        file_path = os.path.join(path, file)
        if os.path.splitext(file_path)[1] == '.png':
            im = Image.open(file_path)
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, im)
            str = os.path.splitext(file_path)[0]+'.jpg'
            bg.save(str)
'''

def listdir(path):
    list_index_name = []
    list_img_name = []
    # png2jpg(path)
    a = os.listdir(path)
    a.sort()

    for file in a:
        file_path = os.path.join(path, file)
        if os.path.splitext(file_path)[1] == '.jpg':
            list_img_name.append(file_path)
        elif os.path.splitext(file_path)[1] == '.csv':
            list_index_name.append(file_path)
    # print(list_img_name[:10])
    # print(list_index_name[:10])
    return list_index_name, list_img_name



# list_index_name, list_img_name = listdir('/home/xinzi/dataset/cartoonset10k')  # 此时两者的路径都是完整的、拼接过的
# a = 0


def read_csv(list_index_name):
    z = []
    for path in list_index_name:
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            column = [row[1] for row in reader]
            i = 0
            for num in column:
                column[i] = int(num)
                i = i + 1
            z.append(column)
    a = 0
    return np.array(z)


#z = read_csv(list_index_name)

'''
def read_img():
    i = 0
    image_arr = np.empty((10000, 3, 500, 500))
    for path in list_img_name[:2]:
        print(path)
        image = Image.open(path)
        # image.show()
        # image.save("/home/xinzi/dataset/1.png")
        image = image.convert("RGB")
        # image.save("/home/xinzi/dataset/2.png")
        # print(image.mode)
        image_arr[i] = np.array(image).transpose((2, 0, 1))  # image_arr大小转换成（3，500, 500）

        #print(image_arr[i].shape)
        i = i + 1
    print(image_arr.shape)

#read_img()
'''


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

'''
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
'''


class MyDataset(data.Dataset):
    def __init__(self, root_path, img_transform=None, img_loader=pil_loader):
        self.path = root_path
        self.transform = img_transform
        self.loader = img_loader
        self.index_path, self.imgs_path = listdir(self.path)
        a = 0

    def __getitem__(self, index):
        z_total = read_csv(self.index_path)
        z = z_total[index]

        path2 = self.imgs_path[index]
        img = self.loader(path2)
        # a = 0
        if self.transform is not None:
            img = self.transform(img)
            # z = self.transform(z)
        return img, z

    def __len__(self):
        return len(self.index_path)



