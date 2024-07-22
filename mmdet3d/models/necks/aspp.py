import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from mmdet.models import NECKS
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer

class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, 
                 conv_layer=nn.Conv2d, bias=False, **kwargs):
        super(Conv, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = conv_layer(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias)
                        
    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.LeakyReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False, conv_layer=conv_layer)

        self.norm = build_norm_layer(norm_cfg, planes)[1]
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(BasicBlock, self).__init__()
        
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size, norm_cfg=norm_cfg)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size, norm_cfg=norm_cfg)
    
        self.act = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out

@NECKS.register_module()
class ASPPNeck(BaseModule):
    def __init__(self, in_channels, out_channels,norm_cfg=None):

        super(ASPPNeck, self).__init__()

        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = dict(type='BN', requires_grad=True),

        self.pre_conv = BasicBlock(in_channels, norm_cfg=self.norm_cfg)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        self.post_conv = ConvBlock(in_channels * 6, out_channels, kernel_size=1, stride=1, norm_cfg=self.norm_cfg)

    def _forward(self, x):
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=18, dilation=18)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        return x

    def forward(self, x):
        if x.requires_grad:
            out = cp.checkpoint(self._forward, x)
        else:
            out = self._forward(x)

        return out