import torch
from torch import Tensor
import torch.nn as nn

class Conv2dBNAct(nn.Module):
    # Standard Conv2d+BN+Act.
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        groups: int = 1,
        has_activation: bool=True,
        act: str="gelu",
        norm: str="batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        gourpnorm_size = 4,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            padding=kernel_size // 2,
            bias=conv_has_bias,
        )

        if norm=="instancenorm":
            self.bn = nn.InstanceNorm2d(out_channels, affine=bn_has_bias, eps=1e-4)
        elif norm=="batchnorm":
            self.bn = nn.BatchNorm2d(out_channels, affine=bn_has_bias,eps=1e-4)
        elif norm=='groupnorm':
            self.bn = nn.GroupNorm(num_groups=out_channels // gourpnorm_size, num_channels=out_channels, affine=bn_has_bias)
        else:
            [print(f"Warning: No normalization used in layer with {out_channels} channels") for _ in range(1)]
            self.bn = nn.Identity()

        if act =='relu':
            act = nn.ReLU()
        elif act =='gelu':
            act = nn.GELU()
        elif act =='silu':
            act = nn.SiLU()
        self.act = act if has_activation is True else nn.Identity()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x: Tensor) -> Tensor: 
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

 
class ResidualConv2dBNAct(nn.Module):
    # Standard Conv2d+BN+Act.
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int,
        groups: int = 1,
        has_activation: bool=True,
        act: str="gelu",
        norm: str="batchnorm",
        conv_has_bias: bool = False,
        bn_has_bias: bool = True,
        gourpnorm_size = 4,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups,
            padding=kernel_size // 2,
            bias=conv_has_bias,
        )

        if in_channels == out_channels and stride==1:
            self._res = nn.Identity()
        else:
            self._res = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=stride,
            groups=1,
            padding=0,
            bias=False,
        )

        if norm=="instancenorm":
            self.bn = nn.InstanceNorm2d(out_channels, affine=bn_has_bias, eps=1e-5)
        elif norm=="batchnorm":
            self.bn = nn.BatchNorm2d(out_channels, affine=bn_has_bias,eps=1e-5)
        elif norm=='groupnorm':
            self.bn = nn.GroupNorm(num_groups=out_channels // gourpnorm_size, num_channels=out_channels, affine=bn_has_bias)
        else:
            self.bn = nn.Identity()

        if act =='relu':
            act = nn.ReLU()
        elif act =='gelu':
            act = nn.GELU()
        elif act =='silu':
            act = nn.SiLU()
        self.act = act if has_activation is True else nn.Identity()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 假设使用ReLU或其变体

    def forward(self, x: Tensor) -> Tensor:
        res_x = self._res(x)

        x = self.conv(x)
        x = self.bn(x)
        
        return  self.act(x + res_x)

class Bottleneck(nn.Module):
    # Stride 1 bottleneck.
    def __init__(self, in_channels: int, expansion_factor: float, has_conv3: bool,act='gelu', norm: str="batchnorm",conv_has_bias=False,bn_has_bias=True):
        super().__init__()
        hidden_channels = int(expansion_factor * in_channels)
        self.conv1 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv2 = Conv2dBNAct(hidden_channels, hidden_channels, 3, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv3 = Conv2dBNAct(hidden_channels, in_channels, 1, 1, has_activation=False, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias) if has_conv3 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions.
    def __init__(self, in_channels: int, out_channels: int,act='gelu', norm: str="batchnorm",conv_has_bias=False,bn_has_bias=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv2 = Conv2dBNAct(in_channels, hidden_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.conv3 = Conv2dBNAct(2 * hidden_channels, out_channels, 1, 1, act=act, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)
        self.bottleneck = Bottleneck(hidden_channels, 1, has_conv3=False, norm=norm,conv_has_bias=conv_has_bias,bn_has_bias=bn_has_bias)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.conv2(x)
        x = self.bottleneck(self.conv1(x))
        x = torch.cat((x, skip), dim=1)
        x = self.conv3(x)
        return x

