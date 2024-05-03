#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-29 18:53:54
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0
""" 实现ViT模型 """

import os
from torch import nn
import torch

class ViT(nn.Module):
    def __init__(self, image_size, image_channels, patch_size, embed_size, num_classes):
        super(ViT, self).__init__()
        self.image_size = image_size    # 图片大小
        self.image_channels = image_channels    # 图像通道数
        self.patch_size = patch_size    # patch大小
        self.embed_size = embed_size    # embedding大小
        self.num_classes = num_classes  # 分类类别数
        
        self.patch_count = (image_size // patch_size) ** 2  # 一个通道中patch的数量
        self.conv = nn.Conv2d(in_channels=image_channels,   # 把每一个patch映射到conv_size维度
                              out_channels=self.patch_size**2, 
                              kernel_size=self.patch_size, 
                              padding=0,
                              stride=self.patch_size)
        self.patch_emb = nn.Linear(self.patch_size**2, embed_size)   # 把conv_size维度映射到embed_size维度
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch_count + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))        
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_size, 
                                                                            nhead=2, 
                                                                            batch_first=True), 
                                                                            num_layers=3)
        self.cls_linear = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        # x:[batch_size, image_channels, image_size, image_size]
        x = self.conv(x)    # [batch_size, conv_size, image_size // patch_size, image_size // patch_size]
        x = x.view(x.size(0), x.size(1), self.patch_count) # [batch_size, conv_size, patch_count]
        x = x.permute(0, 2, 1)  # [batch_size, patch_count, conv_size]
        
        x = self.patch_emb(x)   # [batch_size, patch_count, embed_size]
        
        cls_token = self.cls_token.expand(x.size(0), 1, self.embed_size)  # [batch_size, 1, embed_size]
        
        x = torch.cat([cls_token, x], dim=1)  # [batch_size, patch_count + 1, embed_size]
        x = self.pos_emb + x
        
        y = self.transformer(x)
        return self.cls_linear(y[:, 0, :])  # [batch_size, num_classes]
    
        
if __name__ == '__main__':
    vit = ViT(image_size=28, image_channels=1, patch_size=4, embed_size=16, num_classes=10)
    x = torch.randn(5, 1, 28, 28)
    y = vit(x)
    print(y.shape)

