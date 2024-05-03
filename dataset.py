#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-29 19:27:34
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0
""" 加载Mnist数据集 """
import os

from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, Compose
import torchvision

class Mnist(Dataset):
    def __init__(self, is_train=True):
        super(Mnist, self).__init__()
        self.ds = torchvision.datasets.MNIST('./mnist/',
                                             train=is_train,
                                             download=True)
        self.img_transform = Compose([PILToTensor()])
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img, label = self.ds[index]
        return self.img_transform(img) / 255, label

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    ds = Mnist()
    img, label = ds[0]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()