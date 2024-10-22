import torch
import torch.nn as nn
from mmdet.models import NECKS
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer

    
class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, norm_cfg, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(BasicBlock, self).__init__()
        
        self.block1 = ConvBNRelu(inplanes, inplanes, kernel_size=kernel_size, norm_cfg=norm_cfg)
        self.block2 = ConvBNRelu(inplanes, inplanes, kernel_size=kernel_size, norm_cfg=norm_cfg)
    
        self.act = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out

class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, norm_cfg = dict(type='BN', requires_grad=True)):

        super(ASPPConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = nn.LeakyReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
        
        

class ConvTNormAct(nn.Module):
    def __init__(self, in_planes, out_planes, norm_cfg, kernel_size=2, stride=2, padding=0):
        super(ConvTNormAct, self).__init__()
        
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@NECKS.register_module()
class ASPPNeck(BaseModule):
    def __init__(self, in_channels, out_channels,norm_cfg=None,freeze_layers = False):

        super(ASPPNeck, self).__init__()

        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = dict(type='BN', requires_grad=True)

        self.pre_conv = BasicBlock(in_channels, norm_cfg=self.norm_cfg)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, bias=False, padding=0)
        
        self.post_conv = ConvBNRelu(in_channels * 6, in_channels, kernel_size=1, stride=1, norm_cfg=self.norm_cfg)

        self.branch1 = ASPPConv(in_channels, in_channels, dilation=1, norm_cfg=self.norm_cfg)
        self.branch6 = ASPPConv(in_channels, in_channels, dilation=6, norm_cfg=self.norm_cfg)
        self.branch12 = ASPPConv(in_channels, in_channels, dilation=12, norm_cfg=self.norm_cfg)
        self.branch18 = ASPPConv(in_channels, in_channels, dilation=18, norm_cfg=self.norm_cfg)

        self.upsample_1 = ConvTNormAct(in_channels, out_channels, self.norm_cfg, kernel_size=2, stride=2)

        if freeze_layers:
            for param in self.parameters():
                param.requires_grad = False


    def _forward(self, x):

        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = self.branch1(x)
        branch6 = self.branch6(x)
        branch12 = self.branch12(x)
        branch18 = self.branch18(x)

        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        x = self.upsample_1(x)
        return x

    def forward(self, x):

        x = x[0] # x is a list of last 4 ResNet layers, 0 idx being smallest res

        out = self._forward(x)

        return [out]