# -*- coding: utf-8 -*-
"""
@file      :  FPNEncoder.py
@Time      :  2022/10/14 12:26
@Software  :  PyCharm
@summary   :  THE ENCODER PART OF FPN
@Author    :  Bajian Xiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt


from model.backbone import build_backbone


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, num_blocks, num_classes, back_bone, pretrained=False, ln_out=1):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.back_bone = build_backbone(back_bone)

        # t60 net
        self.resnet_out = 2048
        self.fc1_out = 1024
        self.fc2_out = 512
        self.fc3_out = 256
        self.fc4_out = 128
        self.ln_out = ln_out
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.t60_fc = nn.Sequential(
            nn.Linear(in_features=self.resnet_out, out_features=self.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc1_out),
            nn.Dropout(0.3),

            nn.Linear(in_features=self.fc1_out, out_features=self.fc2_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc2_out),
            nn.Dropout(0.2),

            nn.Linear(in_features=self.fc2_out, out_features=self.fc3_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc3_out),
            nn.Dropout(0.2),

            nn.Linear(in_features=self.fc3_out, out_features=self.fc4_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc4_out),
            nn.Dropout(0.1),

            nn.Linear(in_features=self.fc4_out, out_features=self.ln_out),
        )

    def forward(self, x):
        # Bottom-up using backbone
        # x = [1, 3, 224, 224]
        low_level_features = self.back_bone(x)
        # [1, 2048, 7, 7]
        c5 = low_level_features[4]

        t60 = self.avgpool(c5)      # [1, 2048, 1, 1]
        t60 = torch.flatten(t60, 1) # [1, 2048]
        t60 = self.t60_fc(t60)      # [1, 1]

        return t60

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 这个num_classes实际上是out_channel
    model = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)
    # input = torch.rand(1, 3, 512, 1024)
    input = torch.rand(2, 3, 224, 224)
    t60_out, dereverb_out = model(input) # [1, 3, 224, 224]
    print(dereverb_out.size()) # [1, 32, 224, 224]

