import copy

import torch
import torch.nn as nn

from src.utils.helpers import deep_transform


def SNConv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


def SNLinear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


def SNEmbedding(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))


def net_reparameterize_ops_spectral_norm_pytorch_to_standard(net, inplace=False, net_prefix=None, **kwargs):
    if not inplace:
        net = copy.deepcopy(net)

    def cb_convert(op, prefix, opaque):
        if isinstance(op, torch.nn.Conv2d) or isinstance(op, torch.nn.Linear) or isinstance(op, torch.nn.Embedding):
            try:
                nn.utils.remove_spectral_norm(op)
            except ValueError as e:
                if 'spectral_norm of \'weight\' not found in' in str(e):
                    pass
                else:
                    raise e
        return op

    deep_transform(net, cb_convert, prefix=net_prefix)
    return net
