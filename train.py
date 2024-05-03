#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-29 19:37:29
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import M
from ViT import ViT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

dataset = Mnist()

model = ViT(image_size=28,
            image_channels=1,
            patch_size=4,
            embed_size=16,
            num_classes=10
            ).to(device)

# try:
#     model.load_state_dict(torch.load('model.pth'))
# except:
#     pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
batch_size = 64

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

iter_count = 0
for epoch in range(epochs):
    for imgs, labels in dataloader:
        logits = model(imgs.to(device))

        loss = F.cross_entropy(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_count % 1000 == 0:
            print('epoch:{} iter:{}, loss:{}'.format(epoch, iter_count, loss))
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')
        iter_count += 1

