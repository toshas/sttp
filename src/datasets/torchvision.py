import torch
from torch.utils.data import Dataset
from torch_fidelity.datasets import TransformPILtoRGBTensor

import torchvision
from torchvision import transforms


class AddUniformNoise(object):
    def __init__(self, uniform_low=-1/255., uniform_high=1/255., clamp_low=-1.0, clamp_high=1.0):
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.clamp_low = clamp_low
        self.clamp_high = clamp_high

    def __call__(self, pic):
        if not torch.is_tensor(pic):
            raise TypeError('argument should be a torch tensor. Got {}.'.format(type(pic)))
        if pic.dtype not in (torch.float32, torch.float64):
            raise TypeError('argument should be a floating point tensor. Got {}.'.format(pic.dtype))
        pic += self.uniform_low + torch.rand_like(pic) * (self.uniform_high - self.uniform_low)
        pic.clamp_(self.clamp_low, self.clamp_high)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + f'(uniform: [{self.uniform_low}, {self.uniform_high}], ' \
                                         f'clamp: [{self.clamp_low}, {self.clamp_high}])'


def get_transforms_pil_to_tensor_gan_fmt():
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
        AddUniformNoise(),
    ]


class DropLabelsDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        item = self.ds[index]
        assert type(item) in (tuple, list)
        return item[0]

    def __len__(self):
        return len(self.ds)


def load_cifar10_dataset(root, download, with_labels, evaluation_transforms):
    if evaluation_transforms:
        transform = TransformPILtoRGBTensor()
    else:
        transform = transforms.Compose(get_transforms_pil_to_tensor_gan_fmt())
    dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transform, download=download)
    if not with_labels:
        dataset = DropLabelsDataset(dataset)
    return dataset


def load_stl10_dataset(root, download, with_labels, evaluation_transforms, size=48):
    split = 'train' if with_labels else 'unlabeled'
    if evaluation_transforms:
        transform = TransformPILtoRGBTensor()
    else:
        transform = transforms.Compose([transforms.Resize(size), *get_transforms_pil_to_tensor_gan_fmt()])
    dataset = torchvision.datasets.STL10(root, split=split, transform=transform, download=download)
    if not with_labels:
        dataset = DropLabelsDataset(dataset)
    return dataset
