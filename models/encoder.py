from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.common import Conv2dBNAct, C3,ResidualConv2dBNAct
import torch.nn as nn
import numpy as np
import math

class Backbone(nn.Module):
    def __init__(self, 
                 imgs_color,
                 num_features,
                 d_model,
                 downsampling,
                 num_backbone_layes,
                 cnn_act='gelu',
                 cnn_norm: str="instancenorm",
                 cnn_conv2d_has_bias=False,
                 cnn_bn_has_bias=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        assert imgs_color=="gray" or imgs_color =="rgb" ,"imgs_color should be gray or rgb"
        self.in_channels = 1 if imgs_color == "gray" else 3
        
        # 使用第一个网络的结构定义
        self.block1 = nn.Sequential(
            Conv2dBNAct(self.in_channels, num_features // 2, 5, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
            Conv2dBNAct(num_features//2, num_features, 3, 2, act=cnn_act,norm=cnn_norm),
        )
        num_layers = int(np.log2(downsampling//4))

        self.block2 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 1,act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_backbone_layes)])
        
        self.block3 = nn.ModuleList([Conv2dBNAct(num_features,num_features, 3, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_layers)])
       
        self.block4 = C3(num_features, num_features, act=cnn_act, norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)
        
        self.proj_layer = Conv2dBNAct(num_features,d_model, 1, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)

    def forward(self,x):
        # 使用第二个网络的forward方法逻辑
        res = []
        
        # 分别处理block1中的两个层，以便收集中间特征
        x = self.block1[0](x)  # 第一个Conv2dBNAct层
        x = self.block1[1](x)  # 第二个Conv2dBNAct层
        res.append(x)
        
        for layer in self.block2:
            x = layer(x)
            res.append(x)

        for layer in self.block3:
            x = layer(x)
            res.append(x)

        x = self.block4(x)
        res[-1] = x
        x = self.proj_layer(x)
        
        return x, res[::-1]

# for googlemap and googleerath 128x128
class Backbone_421(nn.Module):
    def __init__(self, 
                 imgs_color,
                 num_features,
                 d_model,
                 downsampling,
                 num_backbone_layes,
                 cnn_act='gelu',
                 cnn_norm: str="batchnorm",
                 cnn_conv2d_has_bias=False,
                 cnn_bn_has_bias=True,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        assert imgs_color in ["gray", "rgb"], "imgs_color should be gray or rgb"
        assert downsampling >= 4 and math.log2(downsampling).is_integer(), "downsampling should be a power of 2 and >= 4"
        
        self.in_channels = 1 if imgs_color == "gray" else 3
        self.downsampling = downsampling

        self.block0 = nn.Sequential(
            Conv2dBNAct(self.in_channels, num_features, 5, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
        )
        
        self.block1 = nn.Sequential(
            ResidualConv2dBNAct(num_features, num_features, 5, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias),
        )

        num_additional_layers = int(math.log2(downsampling//2))

        self.block2 = nn.ModuleList([ResidualConv2dBNAct(num_features, num_features, 3, 2, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias) for _ in range(num_additional_layers)])
       
        self.block3 = C3(num_features, num_features, act=cnn_act)

        self.proj_layer = Conv2dBNAct(num_features,d_model, 1, 1, act=cnn_act,norm=cnn_norm,bn_has_bias=cnn_bn_has_bias,conv_has_bias=cnn_conv2d_has_bias)
        
    def forward(self, x):
        features = []
        
        x = self.block0(x)
        features.append(x)  # 1x

        x = self.block1(x)
        features.append(x)  # 2x

        for layer in self.block2:
            x = layer(x)
            features.append(x)

        x = self.block3(x)
        features[-1] = x  
        x = self.proj_layer(x)
        
        return x, features[-3:][::-1]

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, cnn_norm='group', stride=1):
        super(ResidualBlock, self).__init__()
        """
        contain 2 conv layer, and conv1 invole with stride more than 1
        """

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if cnn_norm == 'groupnorm':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif cnn_norm == 'batchnorm':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif cnn_norm == 'instancenorm':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)

        elif cnn_norm == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), 
            self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    
