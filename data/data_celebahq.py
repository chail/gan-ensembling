import torch
import pandas as pd
import numpy as np
import os
from data.image_dataset import ImageDataset
from torch.utils.data import Subset
from utils import renormalize
from torchvision import transforms
from PIL import Image
import random
from .transforms import randomcrop_tensor, centercrop_tensor


###### utility functions ######

# get the attributes from celebahq subset
def make_table():
    filenames = sorted(os.listdir('dataset/celebahq/images/images'))
    # filter out non-png files, rename it to jpg to match entries
    # in list_attr_celeba.txt
    celebahq = [os.path.basename(f).replace('png', 'jpg')
                for f in filenames if f.endswith('png')]
    attr_gt = pd.read_csv('dataset/celebahq/list_attr_celeba.txt',
                          skiprows=1, delim_whitespace=True, index_col=0)
    # attr_celebahq = attr_gt.loc[celebahq, :].replace(-1, 0)
    attr_celebahq = attr_gt.reindex(index=celebahq).replace(-1, 0)

    # get the train/test/val partitions
    partitions = {}
    with open('dataset/celebahq/list_eval_partition.txt') as f:
        for line in f:
            filename, part = line.strip().split(' ')
            partitions[filename] = int(part)
    partitions_list = [partitions[fname] for fname in attr_celebahq.index]

    attr_celebahq['partition'] = partitions_list
    return attr_celebahq

# make table
attr_celebahq = make_table()

# convert from train/val/test to partition numbers
part_to_int = dict(train=0, val=1, test=2)

def get_partition_indices(part):
    return np.where(attr_celebahq['partition'] == part_to_int[part])[0]


###### dataset functions ######

class CelebAHQDataset:
    def __init__(self, partition, attribute, load_w=True, fraction=None,
                 **kwargs):
        root = 'dataset/celebahq/images'
        self.load_w = load_w
        self.fraction = fraction
        if self.load_w:
            # get image path as well
            self.dset = ImageDataset(root, return_path=True, **kwargs)
        else:
            self.dset = ImageDataset(root, **kwargs)
        partition_idx = get_partition_indices(partition)
        # if we want to further subsample the dataset, just subsample
        # partition_idx and Subset() once
        if fraction is not None:
            print("Using a fraction of the original dataset")
            print("The original dataset has length %d" % len(partition_idx))
            new_length = int(fraction / 100  * len(partition_idx))
            rng = np.random.RandomState(1)
            new_indices = rng.choice(partition_idx, new_length, replace=False)
            partition_idx = new_indices
            print("The subsetted dataset has length %d" % len(partition_idx))

        self.dset = Subset(self.dset, partition_idx)
        attr_subset = attr_celebahq.iloc[partition_idx]
        self.attr_subset = attr_subset[attribute]
        print('attribute freq: %0.4f (%d / %d)' % (self.attr_subset.mean(),
                                                   self.attr_subset.sum(),
                                                   len(self.attr_subset)))

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data = self.dset[idx]
        # first element is the class, replace it
        label = self.attr_subset[idx]
        if self.load_w:
            path = data[2]
            w_path = path.replace('images/images', 'latents').replace('png', 'pth')
            # use [0] to remove the batch dimension
            w_pth = torch.load(w_path, map_location='cpu')['w'][0].detach()

            return (data[0], w_pth, label, *data[2:])
        return (data[0], label, *data[2:])


class CelebAHQIDInvertDataset(CelebAHQDataset):
    def __getitem__(self, idx):
        data = self.dset[idx]
        # first element is the class, replace it
        label = self.attr_subset[idx]
        # sanity check: check the size of the image; it should be 256
        if type(data[0]) == torch.Tensor:
            assert(data[0].shape[-1] == 256)
        else: # Image file
            assert(data[0].size[-1] == 256)
        if self.load_w:
            path = data[2]
            w_path = path.replace('images/images', 'latents_idinvert').replace('png', 'npy')
            w_pth = torch.from_numpy(np.load(w_path))[0] # remove batch dim
            return (data[0], w_pth, label, *data[2:])
        return (data[0], label, *data[2:])


###### transformation functions ######

def get_transform(dataset, transform_type):
    if dataset.lower() == "celebahq":
        # official stylegan output size is 1024
        base_size = 1024
    elif dataset.lower() == "celebahq-idinvert":
        # gan output size is 256
        base_size = 256

    if transform_type == 'imtrain':
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'imval':
        return transforms.Compose([
            transforms.Resize(base_size),
            # no horizontal flip for standard validation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'imcolor':
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=.05, contrast=.05,
                                   saturation=.05, hue=.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_type == 'imcrop':
        return transforms.Compose([
            # 1024 + 32, or 256 + 8
            transforms.Resize(int(1.03125 * base_size)),
            transforms.RandomCrop(base_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_type == 'tensorbase':
        # dummy transform for compatibility with other datasets
        return transforms.Lambda(lambda x: x)
    elif transform_type == 'tensormixed':
        return TensorCombinedTransform(base_size)
    else:
        raise NotImplementedError

class TensorCombinedTransform(object):
    def __init__(self, base_size, horizontal_flip=True, color_jitter=True):
        if base_size == 1024:
            self.load_size = 1056
            self.crop_size = 1024
        elif base_size == 256:
            self.load_size = 264
            self.crop_size = 256
        self.flip = horizontal_flip
        self.color_jitter = color_jitter
        self.resize = torch.nn.Upsample(self.load_size, mode='bilinear')
        self.colorjitter = transforms.ColorJitter(brightness=.05, contrast=.05,
                                                  saturation=.05)

    def __call__(self, image):
        # resize it
        image = self.resize(image)
        # perform a random crop for each image in the batch individually
        image = torch.cat([randomcrop_tensor(im[None], self.crop_size) for im in image])
        # if needed, add flip for each image in the batch individually
        if self.flip:
            image = torch.stack([torch.flip(x, dims=(-1,)) if torch.rand(1) >
                                 0.5 else x for x in image])
        # if needed, add color jitter for each image in the batch individually
        if self.color_jitter:
            image = renormalize.as_tensor(image, source='zc', target='pt')
            image = torch.stack([self.colorjitter(x) for x in image])
            image = renormalize.as_tensor(image, source='pt', target='zc')
        return image
