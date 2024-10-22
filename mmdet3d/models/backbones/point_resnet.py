import torch.nn as nn
from ..builder import BACKBONES
from mmcv.cnn import build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_cfg=None):
        super(BasicBlock, self).__init__()
        if norm_cfg is None:
            norm_layer = nn.BatchNorm2d(planes)
        else:
            norm_layer = build_norm_layer(norm_cfg, planes)[1]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer
        self.downsample = downsample
        self.stride = stride

        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class PointResNet34V2(nn.Module):
    """
    ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Adjusted for use as a 3D backbone, so different block structure according to FastPillars: https://arxiv.org/abs/2302.02367
    """
    def __init__(self, block= BasicBlock, layers=[6,6,3,1], in_channels = 64,
                 groups=1, width_per_group=64,
                 norm_cfg=None, freeze_layers = False):
        super(PointResNet34V2, self).__init__()
        
        self.norm_cfg = norm_cfg
        self.freeze = freeze_layers

        self.inplanes = 64

        if norm_cfg is None:
            norm_layer = nn.BatchNorm2d(self.inplanes)
        else:
            norm_layer = build_norm_layer(norm_cfg, self.inplanes)[1]


        self.dilation = 1
        self.in_channels = in_channels
        
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer
        self.relu = nn.LeakyReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.freeze:
            print("Freeze PointResNet34 layers")
            # Freeze all layers
            self.freeze_layers()

    def freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        previous_dilation = self.dilation
 
        if stride != 1 or self.inplanes != planes * block.expansion:

            norm_layer = build_norm_layer(self.norm_cfg, planes * block.expansion)[1]

            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_cfg= self.norm_cfg))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        stage_1_out = self.layer1(x)
        stage_2_out = self.layer2(stage_1_out)
        stage_3_out = self.layer3(stage_2_out)
        stage_4_out = self.layer4(stage_3_out)

        return [stage_4_out, stage_3_out]

    def forward(self, x):
        return self._forward_impl(x)