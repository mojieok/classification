import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        #每个类别的图片分别存放在不同的文件夹中，并且该文件夹名就是标签名
        self.label_name = {"ants": 0, "bees": 1}#获取标签名称
        self.data_info = self.get_img_info(data_dir)#data_info是一个List,里边存放图片的位置以及标签
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)#返回数据集的样本总数

    def get_img_info(self, data_dir):#data_dir是数据所在文件夹
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):#遍历数据所在文件夹
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            #判断data_dir文件夹中是否有图片，如果没有就抛出异常
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info
