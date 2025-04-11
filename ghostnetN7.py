#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['ghostnetN7']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


#This function was introduced in the author's code. It is an approximation of the sigmoid function,
#removing the complexities of the exponential component, thus making its application more lightweight.
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

#Rather than using the Author's implementation of Squeeze and Excite, which only uses channel attention, I have decided to reimplement this
#program using Convolutional Block Attention Model (CBAM) which adds spacial attention as well:
#Important Changes Made:
#1)I have chosen to include spacial attention, so I include a kernel attribute which I will pass through in GhostNet
#2)I have chosen to keep the channel attribute ratio the same in order to keep the computational cost down as I have already added a spacial component.
class CBAM(nn.Module):
    def __init__(self, input_channels, cbam_ratio=0.25, reduced_base_chs=None, spatial_kernel = 7,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):

        super(CBAM, self).__init__()
        self.gate_fn = gate_fn  # Activation function for attention, using Hard_sigmoid to reduce complexity.

        # Compute reduced channels for Channel Attention
        reduced_channels = _make_divisible((reduced_base_chs or input_channels) * cbam_ratio, divisor)

        # Channel Attention Module (CAM)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_reduce = nn.Conv2d(input_channels, reduced_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, input_channels, 1, bias=True)

        # Spatial Attention Module (SAM)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        # Channel Attention
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_ca = self.conv_reduce(x_avg) + self.conv_reduce(x_max)  # Shared MLP
        x_ca = self.act1(x_ca)
        x_ca = self.conv_expand(x_ca)
        x = x * self.gate_fn(x_ca)  # Appling the Channel Attention

        # Spatial Attention
        x_avg_sp = torch.mean(x, dim=1, keepdim=True)  # Avg Pool along channel axis
        x_max_sp, _ = torch.max(x, dim=1, keepdim=True)  # Max Pool along channel axis
        x_sa = torch.cat([x_avg_sp, x_max_sp], dim=1)  # Concatenate along channel dim
        x_sa = self.conv_spatial(x_sa)
        x = x * self.gate_fn(x_sa)  # Appling the Spatial Attention
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=7, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        #I want to add additional code to the ghost module to merge the primary convolutions with the first layer of ghost operations:

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck with CBAM Implementation"""

    def __init__(self, input_channels, mid_channels, out_channels, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, cbam_ratio=0., spatial_kernel = 7):

        super(GhostBottleneck, self).__init__()
        has_cbam = cbam_ratio is not None and cbam_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(input_channels, mid_channels, relu=True)

        # Depth-wise convolution (Ignored unless stride is called on to be greater)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        # Convolutional Block Attention Model (CBAM) Activation:
        #For initial implementation, I would like to test the spacial kernel size to be 7
        if has_cbam:
            self.cbam = CBAM(mid_channels, cbam_ratio=cbam_ratio, spatial_kernel = spatial_kernel)
        else:
            self.cbam = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # shortcut
        if (input_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, dw_kernel_size, stride=stride, padding=(dw_kernel_size-1)//2, groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.Conv2d(input_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # CBAM
        if self.cbam is not None:
            x = self.cbam(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet_N7(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):

        super(GhostNet_N7, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        #Building First Layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        #I have added a spacial_kernel component to the architecture to input different sized spacial attention filters during the network.
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, cbam_ratio, s, spatial_kernel in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, cbam_ratio=cbam_ratio, spatial_kernel = spatial_kernel))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnetN7(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k = kernel size, t = expansion factor , c = # of Feature Maps, CBAM = cbam ratio, s = stride, sk = spatial kernel
        # stage1
        [[3,  16,  16, 0, 1,0]],  #The Spatial Kernel Size is set to 0 when CBAM is not active for a particular layer of the GhostNet
        # stage2
        [[3,  48,  24, 0, 2,0]],
        [[3,  72,  24, 0, 1,0]],
        # stage3
        [[5,  72,  40, 0.25, 2,7]], #Keep the default spacial attention for intitial layers to a kernel size of 7
        [[5, 120,  40, 0.25, 1,7]],
        # stage4
        [[3, 240,  80, 0, 2,0]],
        [[3, 200,  80, 0, 1,0],
         [3, 184,  80, 0, 1,0],
         [3, 184,  80, 0, 1,0],
         [3, 480, 112, 0.25, 1,5],  #Increase the Spacial Attention at the end layers to 5
         [3, 672, 112, 0.25, 1,5]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2, 3]], #The final layer will get the most extensive spacial attention with a kernel size of 3. (This can be changed if complexity is increased too much)
        [[5, 960, 160, 0, 1,0],
         [5, 960, 160, 0.25, 1,3],
         [5, 960, 160, 0, 1,0],
         [5, 960, 160, 0.25, 1,3]
        ]
    ]
    return GhostNet_N8(cfgs, **kwargs)


if __name__=='__main__':
    model = ghostnetN7()
    model.eval()
    print(model)
    input = torch.randn(32,3,320,256)
    y = model(input)
    print(y.size())

