from mmdet.models import NECKS
import numpy as np
from mmcv.cnn import xavier_init
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from torch import nn
from mmcv.runner import force_fp32, auto_fp16

import inspect
import sys
from collections import OrderedDict

import numpy as np
import torch


class ConvTBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, norm_cfg=None):
        super(ConvTBNReLU, self).__init__()

        print(f"padding in ConvTBNRelu:{padding}")
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_planes)[1]
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_cfg=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        
        # Define a shortcut layer for the residual connection
        self.shortcut = nn.Identity()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, planes)[1]
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

@NECKS.register_module()
class RPNV2(BaseModule):
    def __init__(
         self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        freeze_layers = False
    ):
        super(RPNV2, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features
        self.freeze = freeze_layers

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.block_5, num_out_filters = self._make_layer(
            self._num_input_features[1],
            self._num_filters[1],
            self._layer_nums[1],
            stride=1,
        )
       
        norm_deblock5 = build_norm_layer(self._norm_cfg, self._num_upsample_filters[1])[1]
        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(
                num_out_filters,
                self._num_upsample_filters[1],
                2,
                stride=2,
                bias=False,
            ),
            norm_deblock5,
            nn.LeakyReLU(),
        )
        norm_deblock4 = build_norm_layer(self._norm_cfg, self._num_upsample_filters[0])[1]
        
        self.deblock_4 = Sequential(
            nn.Conv2d(self._num_input_features[0], self._num_upsample_filters[0], 3, stride=1, padding=1, bias=False),
            norm_deblock4,
            nn.LeakyReLU(),
        )
        self.block_4, num_out_filters = self._make_layer(
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._layer_nums[0],
            stride=1,
        )

        self.additional_upsample = ConvTBNReLU(num_out_filters, num_out_filters, kernel_size=2, stride=2, padding=0, norm_cfg=self._norm_cfg)

        if self.freeze:
            # Freeze all layers
            self.freeze_layers()

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        layers = []

        # Initial Conv Layer
        layers.append(nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False))
        layers.append(build_norm_layer(self._norm_cfg, planes)[1])
        layers.append(nn.LeakyReLU())

        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(planes, planes, stride, self._norm_cfg))

        return nn.Sequential(*layers), planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def freeze_layers(self):
        print("Freeze neck layers")
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, **kwargs):
        
        #get feature maps of last 2 resolutions 
        x_conv4 = x[1] 
        x_conv5 = x[0]

        ups = [self.deblock_4(x_conv4)]
        x = self.block_5(x_conv5)


        ups.append(self.deblock_5(x))

        x = torch.cat(ups, dim=1)
        x = self.block_4(x)
        x = self.additional_upsample(x)

        return [x]
    
@NECKS.register_module()
class RPNV3(BaseModule):
    def __init__(
         self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        freeze_layers = False
    ):
        super(RPNV3, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features
        self.freeze = freeze_layers

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.block_5, num_out_filters = self._make_layer(
            self._num_input_features[1],
            self._num_filters[1],
            self._layer_nums[1],
            stride=1,
        )
       
        norm_deblock5 = build_norm_layer(self._norm_cfg, self._num_upsample_filters[1])[1]
        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(
                num_out_filters,
                self._num_upsample_filters[1],
                2,
                stride=2,
                bias=False,
            ),
            norm_deblock5,
            nn.LeakyReLU(),
        )
        norm_deblock4 = build_norm_layer(self._norm_cfg, self._num_upsample_filters[0])[1]
        
        self.deblock_4 = Sequential(
            nn.Conv2d(self._num_input_features[0], self._num_upsample_filters[0], 3, stride=1, padding=1, bias=False),
            norm_deblock4,
            nn.LeakyReLU(),
        )
        self.block_4, num_out_filters = self._make_layer(
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._num_upsample_filters[0] + self._num_upsample_filters[1],
            self._layer_nums[0],
            stride=1,
        )

        if self.freeze:
            # Freeze all layers
            self.freeze_layers()

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        layers = []

        # Initial Conv Layer
        layers.append(nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False))
        layers.append(build_norm_layer(self._norm_cfg, planes)[1])
        layers.append(nn.LeakyReLU())

        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(planes, planes, stride, self._norm_cfg))

        return nn.Sequential(*layers), planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def freeze_layers(self):
        print("Freeze neck layers")
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, **kwargs):
        
        #get feature maps of last 2 resolutions 
        x_conv4 = x[1] 
        x_conv5 = x[0]

        ups = [self.deblock_4(x_conv4)]
        x = self.block_5(x_conv5)


        ups.append(self.deblock_5(x))

        x = torch.cat(ups, dim=1)
        x = self.block_4(x)

        return [x]
    

class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.LeakyReLU(),
                  nn.Conv2d(20,64,5),
                  nn.LeakyReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.LeakyReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.LeakyReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.LeakyReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.LeakyReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
        )


class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """

    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(
                f"{msg} shape: {tensor.shape}"
                f" max: {tensor.max()} min: {tensor.min()}"
                f" mean: {tensor.mean()}"
            )

    return printer


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator
