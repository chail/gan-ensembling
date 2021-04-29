import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np
import torch

class CIFAR10Dataset:
    def __init__(self, partition, load_w=True, **kwargs):
        self.load_w = load_w
        self.partition = partition
        if partition == 'train':
            trainset = torchvision.datasets.CIFAR10(
                root='./dataset/cifar10', train=True, download=True,
                **kwargs)
            self.dset = Subset(trainset, range(45000))
            self.subset_indices = list(range(45000))
        elif partition == 'val':
            trainset = torchvision.datasets.CIFAR10(
                root='./dataset/cifar10', train=True, download=True,
                **kwargs)
            self.dset = Subset(trainset, range(45000, 50000))
            self.subset_indices = list(range(45000, 50000))
        elif partition == 'test':
            testset = torchvision.datasets.CIFAR10(
                    root='./dataset/cifar10', train=False, download=True, **kwargs)
            self.dset = testset
            self.subset_indices = list(range(10000))

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data = self.dset[idx]
        if self.load_w:
            w_path = ('dataset/cifar10/latents/%s_%06d.npy'%
                      (self.partition, idx))
            # remove batch dim
            opt_w = torch.from_numpy(np.load(w_path))[0]
            return (data[0], opt_w, *data[1:])
        else:
            return data

###### transformation functions ######

def get_transform(dataset, transform_type):
    assert(dataset.lower() == 'cifar10')
    if transform_type == 'imtrain':
        return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_type == 'imval':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'tensorbase':
        # dummy transform for compatibility with other datasets
        return transforms.Lambda(lambda x: x)
    elif transform_type == 'tensortrain':
        return TensorTransformTrain()

class TensorTransformTrain(object):
    def __init__(self):
        # fill changed to -1 bc tensor is zero centered
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=-1),
            transforms.RandomHorizontalFlip()
        ])

    def __call__(self, tensor):
        # applies the transform for each image in the batch individually
        return torch.stack([self.transform(t) for t in tensor])
