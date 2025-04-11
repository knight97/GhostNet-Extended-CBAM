#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dw_size=3, ratio=2, stride=1, padding=0, bias=False):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.init_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channels, self.new_channels, dw_size, stride=1,
                      padding=dw_size // 2, groups=self.init_channels, bias=bias),
            nn.BatchNorm2d(self.new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        if self.new_channels == 0:
            return x1
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :x1.size(1) * self.ratio, :, :]


def conv3x3(in_planes, out_planes, stride=1, s=4, d=3):
    return GhostModule(in_planes, out_planes, kernel_size=3, dw_size=d, ratio=s,
                       stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(Bottleneck, self).__init__()
        self.conv1 = GhostModule(inplanes, planes, kernel_size=1, dw_size=d, ratio=s)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s, stride=stride, padding=1)
        self.conv3 = GhostModule(planes, planes * 4, kernel_size=1, dw_size=d, ratio=s)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, s=4, d=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, s=s, d=d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, s=s, d=d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, s=s, d=d)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, s=s, d=d)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, s=4, d=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                GhostModule(self.inplanes, planes * block.expansion, ratio=s, dw_size=d, kernel_size=1, stride=stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, s=s, d=d))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, s=s, d=d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

