#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-29 19:53:51
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0

import os
from dataset import Mnist
import matplotlib.pyplot as plt 
import torch 
from ViT import ViT
import torch.nn.functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# 加载测试集
dataset = Mnist()

model = ViT(image_size=28, image_channels=1, patch_size=4, embed_size=16, num_classes=10).to(device)
model.load_state_dict(torch.load('model.pth'))

model.eval()

image, label = dataset[5]
logits = model(image.unsqueeze(0).to(device))
print("正确类别：", label)
print("预测类别：", torch.argmax(logits, dim=-1).item())
plt.imshow(image.permute(1, 2, 0))
plt.show()



