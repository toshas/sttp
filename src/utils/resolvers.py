from functools import partial

import torch
import torchvision
from torch.optim import SGD, Adam

from src.datasets.torchvision import load_cifar10_dataset, load_stl10_dataset
from src.modules import ops_spectral_norm_pytorch
from src.modules.losses import hinge_loss_gen, hinge_loss_dis
from src.modules.model_resnet_cifar10 import ModelNetResnetCifar10
from src.modules.model_wideresnet_cifar import ModelNetWresnetCifar
from src.modules.sngan import SNGANGenerator, SNGANDiscriminator
from src.utils.spectral_penalties import spectral_penalty_d_optimal, spectral_penalty_divergence
from src.utils.spectral_tensors_factory import SpectralTensorsFactorySVDP, SpectralTensorsFactorySTTP
from src.utils.stiefel_parameterization import StiefelHouseholder, StiefelHouseholderCanonical


def resolve_optimizer(name):
    return {
        'sgd': SGD,
        'adam': Adam,
    }[name]


def resolve_lr_sched(optimizer, name, num_training_steps):
    if name == 'linear':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0, num_training_steps - step) / num_training_steps
        )
    else:
        raise ValueError(f'LR scheduler {name} not supported')


def resolve_gan_models(name):
    return {
        'sngan': (SNGANGenerator, SNGANDiscriminator),
    }[name]


def resolve_ops(name):
    return {
        'regular': {
            'cls_conv2d': torch.nn.Conv2d,
            'cls_linear': torch.nn.Linear,
            'cls_embedding': torch.nn.Embedding,
        },
        'spectral_norm_pytorch': {
            'cls_conv2d': ops_spectral_norm_pytorch.SNConv2d,
            'cls_linear': ops_spectral_norm_pytorch.SNLinear,
            'cls_embedding': ops_spectral_norm_pytorch.SNEmbedding,
        },
    }[name], \
    {
        'regular': None,
        'spectral_norm_pytorch': ops_spectral_norm_pytorch.net_reparameterize_ops_spectral_norm_pytorch_to_standard,
    }[name]


def resolve_ops_factory(name):
    return {
        'svdp': SpectralTensorsFactorySVDP,
        'sttp': SpectralTensorsFactorySTTP,
    }[name]


def resolve_gan_losses(name):
    return {
        'hinge': (hinge_loss_gen, hinge_loss_dis),
    }[name]


def resolve_gan_dataset(name, root, download, with_labels, evaluation_transforms):
    return {
        'cifar10': load_cifar10_dataset,
        'stl10_48': partial(load_stl10_dataset, size=48),
    }[name](root, download, with_labels, evaluation_transforms), \
    {
        'cifar10': 10,
        'stl10_48': 10,
    }[name]


def resolve_spectral_penalty(name):
    return {
        'd_optimal': spectral_penalty_d_optimal,
        'divergence': spectral_penalty_divergence,
    }[name]


def resolve_stiefel(name, is_canonical):
    assert name == 'householder'
    if is_canonical:
        return StiefelHouseholderCanonical
    else:
        return StiefelHouseholder


def resolve_imgcls_dataset(cfg):
    transform_train, transform_valid = [], []
    if cfg.dataset in ('cifar10',):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = {
            'cifar10': 10,
        }[cfg.dataset]
        transform_train += [
            torchvision.transforms.Pad(4),
            torchvision.transforms.RandomResizedCrop(32),
        ]
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')

    transform_train += [
        torchvision.transforms.RandomHorizontalFlip()
    ]
    transform_epilogue = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
    transform_train += transform_epilogue
    transform_valid += transform_epilogue

    transform_train = torchvision.transforms.Compose(transform_train)
    transform_valid = torchvision.transforms.Compose(transform_valid)

    if cfg.dataset in ('cifar10',):
        dataset_class = {
            'cifar10': torchvision.datasets.CIFAR10,
        }[cfg.dataset]
        dataset_train = dataset_class(
            cfg.root_datasets[cfg.dataset], train=True, transform=transform_train, download=cfg.dataset_download
        )
        dataset_valid = dataset_class(
            cfg.root_datasets[cfg.dataset], train=False, transform=transform_valid, download=cfg.dataset_download
        )
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')

    return dataset_train, dataset_valid, num_classes


def resolve_imgcls_model(name):
    return {
        'resnet_cifar10': ModelNetResnetCifar10,
        'wresnet_cifar': ModelNetWresnetCifar,
    }[name]

