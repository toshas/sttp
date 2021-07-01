import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    32: {  # CIFAR-10
        'nz': 128,  # Noise dimension
        'ngf': 256,  # Number of generator features
        'ndf': 128,  # Number of discriminator features
        'bottom_width': 4,  # spatial size of the first convolutional input
        'g_blocks': [  # {in|out}_channels = {ngf|ndf} // {in|out}_factor
            {'in_factor': 1, 'out_factor': 1, 'upsample': True},
            {'in_factor': 1, 'out_factor': 1, 'upsample': True},
            {'in_factor': 1, 'out_factor': 1, 'upsample': True},
        ],
        'd_blocks': [
            {'in_factor': 1, 'out_factor': 1, 'downsample': True},
            {'in_factor': 1, 'out_factor': 1, 'downsample': False},
            {'in_factor': 1, 'out_factor': 1, 'downsample': False},
        ]
    },
    48: {  # STL-10
        'nz': 128,
        'ngf': 512,
        'ndf': 1024,
        'bottom_width': 6,
        'g_blocks': [
            {'in_factor': 1, 'out_factor': 2, 'upsample': True},
            {'in_factor': 2, 'out_factor': 4, 'upsample': True},
            {'in_factor': 4, 'out_factor': 8, 'upsample': True},
        ],
        'd_blocks': [
            {'in_factor': 16, 'out_factor': 8, 'downsample': True},
            {'in_factor': 8, 'out_factor': 4, 'downsample': True},
            {'in_factor': 4, 'out_factor': 2, 'downsample': True},
            {'in_factor': 2, 'out_factor': 1, 'downsample': False},
        ]
    },
    64: {  # https://github.com/pfnet-research/sngan_projection
        'nz': 128,
        'ngf': 1024,
        'ndf': 1024,
        'bottom_width': 4,
        'g_blocks': [
            {'in_factor': 1, 'out_factor': 2, 'upsample': True},
            {'in_factor': 2, 'out_factor': 4, 'upsample': True},
            {'in_factor': 4, 'out_factor': 8, 'upsample': True},
            {'in_factor': 8, 'out_factor': 16, 'upsample': True},
        ],
        'd_blocks': [
            {'in_factor': 16, 'out_factor': 8, 'downsample': True},
            {'in_factor': 8, 'out_factor': 4, 'downsample': True},
            {'in_factor': 4, 'out_factor': 2, 'downsample': True},
            {'in_factor': 2, 'out_factor': 1, 'downsample': True},
        ]
    },
    128: {  # ImageNet
        'nz': 128,
        'ngf': 1024,
        'ndf': 1024,
        'bottom_width': 4,
        'g_blocks': [
            {'in_factor': 1, 'out_factor': 1, 'upsample': True},
            {'in_factor': 1, 'out_factor': 2, 'upsample': True},
            {'in_factor': 2, 'out_factor': 4, 'upsample': True},
            {'in_factor': 4, 'out_factor': 8, 'upsample': True},
            {'in_factor': 8, 'out_factor': 16, 'upsample': True},
        ],
        'd_blocks': [
            {'in_factor': 16, 'out_factor': 8, 'downsample': True},
            {'in_factor': 8, 'out_factor': 4, 'downsample': True},
            {'in_factor': 4, 'out_factor': 2, 'downsample': True},
            {'in_factor': 2, 'out_factor': 1, 'downsample': True},
            {'in_factor': 1, 'out_factor': 1, 'downsample': False},
        ]
    },
}


def xavier_init_uniform_if_amenable(op, value):
    if hasattr(op, 'weight') and type(op.weight) is torch.nn.parameter.Parameter:
        nn.init.xavier_uniform_(op.weight.data, value)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        torch.nn.init.ones_(self.gamma.weight)
        torch.nn.init.zeros_(self.beta.weight)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y)
        beta = self.beta(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            upsample=False,
            num_classes=0,
            cls_conv2d=None,
    ):
        super().__init__()
        assert cls_conv2d is not None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_shortcut = in_channels != out_channels or upsample
        self.upsample = upsample

        self.num_classes = num_classes

        self.c1 = cls_conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = cls_conv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)

        self.activation = nn.ReLU(True)

        xavier_init_uniform_if_amenable(self.c1, math.sqrt(2.0))
        xavier_init_uniform_if_amenable(self.c2, math.sqrt(2.0))

        if self.learnable_shortcut:
            self.c_shortcut = cls_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            xavier_init_uniform_if_amenable(self.c_shortcut, 1.0)

    def _upsample_conv(self, x, conv):
        return conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

    def _residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _residual_conditional(self, x, y):
        h = x
        h = self.b1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _shortcut(self, x):
        if self.learnable_shortcut:
            x = self._upsample_conv(x, self.c_shortcut) if self.upsample else self.c_shortcut(x)
        return x

    def forward(self, x, y=None):
        if y is None:
            return self._residual(x) + self._shortcut(x)
        else:
            return self._residual_conditional(x, y) + self._shortcut(x)


class DBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            downsample=False,
            cls_conv2d=None,
    ):
        super().__init__()
        assert cls_conv2d is not None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_shortcut = (in_channels != out_channels) or downsample

        self.c1 = cls_conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = cls_conv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU(True)

        xavier_init_uniform_if_amenable(self.c1, math.sqrt(2.0))
        xavier_init_uniform_if_amenable(self.c2, math.sqrt(2.0))

        if self.learnable_shortcut:
            self.c_shortcut = cls_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            xavier_init_uniform_if_amenable(self.c_shortcut, 1.0)

    def _residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        if self.learnable_shortcut:
            x = self.c_shortcut(x)
            return F.avg_pool2d(x, 2) if self.downsample else x
        return x

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    def __init__(self, in_channels, out_channels, cls_conv2d=None):
        super().__init__()
        assert cls_conv2d is not None
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c1 = cls_conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = cls_conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.c_shortcut = cls_conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = nn.ReLU(True)

        xavier_init_uniform_if_amenable(self.c1, math.sqrt(2.0))
        xavier_init_uniform_if_amenable(self.c2, math.sqrt(2.0))
        xavier_init_uniform_if_amenable(self.c_shortcut, 1.0)

    def _residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        return self.c_shortcut(F.avg_pool2d(x, 2))

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class SNGANGenerator(torch.nn.Module):
    def __init__(self, preset_name, special_ops, num_classes=0, override_ngf=None):
        super().__init__()
        self.num_classes = num_classes

        preset = PRESETS[preset_name]
        self.nz = preset['nz']
        ngf = preset['ngf'] if override_ngf is None else override_ngf
        bottom_width = preset['bottom_width']
        self.bottom_width = bottom_width

        self.noise_to_conv = nn.Linear(self.nz, (bottom_width ** 2) * ngf)
        self.blocks = torch.nn.ModuleList([
            GBlock(
                in_channels=ngf // b['in_factor'],
                out_channels=ngf // b['out_factor'],
                upsample=b['upsample'],
                num_classes=num_classes,
                cls_conv2d=special_ops['cls_conv2d'],
            ) for b in preset['g_blocks']
        ])

        last_num_features = ngf // preset['g_blocks'][-1]['out_factor']

        self.last_bn = nn.BatchNorm2d(last_num_features)
        self.last_conv = nn.Conv2d(last_num_features, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        xavier_init_uniform_if_amenable(self.noise_to_conv, 1.0)
        xavier_init_uniform_if_amenable(self.last_conv, 1.0)

    def forward(self, *args):
        if self.num_classes > 0:
            assert len(args) == 2 and all(torch.is_tensor(a) for a in args)
            x, y = args
        else:
            assert len(args) in (1, 2) and torch.is_tensor(args[0])
            assert len(args) == 1 or (torch.is_tensor(args[1]) or args[1] is None)
            if len(args) == 1:
                x, y = args[0], None
            else:
                x, y = args

        h = self.noise_to_conv(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        for block in self.blocks:
            h = block(h, y)
        h = self.last_bn(h)
        h = self.activation(h)
        h = self.last_conv(h).tanh()

        if not self.training:
            h = (255 * (h.clamp(-1, 1) * 0.5 + 0.5)).to(torch.uint8)

        return h

    @property
    def z_sz(self):
        return self.nz


class SNGANDiscriminator(torch.nn.Module):
    def __init__(self, preset_name, special_ops, num_classes=0, override_ndf=None):
        super().__init__()
        self.num_classes = num_classes

        preset = PRESETS[preset_name]
        ndf = preset['ndf'] if override_ndf is None else override_ndf
        cls_conv2d = special_ops['cls_conv2d']
        cls_linear = special_ops['cls_linear']
        cls_embedding = special_ops['cls_embedding']

        self.block1 = DBlockOptimized(3, ndf // preset['d_blocks'][0]['in_factor'], cls_conv2d=cls_conv2d)
        self.blocks = torch.nn.ModuleList([
            DBlock(
                in_channels=ndf // b['in_factor'],
                out_channels=ndf // b['out_factor'],
                downsample=b['downsample'],
                cls_conv2d=cls_conv2d,
            )
            for b in preset['d_blocks']
        ])

        self.activation = nn.ReLU(True)

        self.last_linear = cls_linear(ndf, 1, bias=False)
        if num_classes > 0:
            self.cond_projection = cls_embedding(num_classes, ndf)
            xavier_init_uniform_if_amenable(self.cond_projection, 1.0)

        xavier_init_uniform_if_amenable(self.last_linear, 1.0)

    def forward(self, *args):
        if self.num_classes > 0:
            assert len(args) == 2 and all(torch.is_tensor(a) for a in args)
            x, y = args
        else:
            assert len(args) in (1, 2) and torch.is_tensor(args[0])
            assert len(args) == 1 or (torch.is_tensor(args[1]) or args[1] is None)
            if len(args) == 1:
                x, y = args[0], None
            else:
                x, y = args

        h = x
        h = self.block1(h)
        for block in self.blocks:
            h = block(h)
        h = self.activation(h)

        h = torch.sum(h, dim=(2, 3))
        output = self.last_linear(h)

        if self.num_classes > 0:
            w_y = self.cond_projection(y)
            output += torch.sum((w_y * h), dim=1, keepdim=True)

        return output
