import torch
import pandas as pd
import numpy as np
import os
from data.image_dataset import ImageDataset
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image
import random
import math

from collections import defaultdict, OrderedDict
from .transforms import randomcrop_tensor, centercrop_tensor


###### utility functions ######
def get_partition_indices_pets(part, dset):
    filenames = [os.path.basename(x[0]) for x in dset.samples]
    indices_per_class = defaultdict(list)
    for i, f in enumerate(filenames):
        if f[0].isupper(): # cats have uppercase first letter
            label = f.split('_')[0]
            indices_per_class[label].append(i)
    train_indices = {k: v[:len(v)//2] for k, v in indices_per_class.items()}
    val_indices = {k: v[len(v)//2:-len(v)//4] for k, v in indices_per_class.items()}
    test_indices = {k: v[-len(v)//4:] for k, v in indices_per_class.items()}
    label_map = OrderedDict([(k, i) for i, k in enumerate(indices_per_class)])
    coarse_labels = label_map.keys()
    partition_indices = []
    if part == 'train':
        [partition_indices.extend(train_indices[k]) for k in train_indices]
    if part == 'val':
        [partition_indices.extend(val_indices[k]) for k in val_indices]
    if part == 'test':
        [partition_indices.extend(test_indices[k]) for k in test_indices]
    return partition_indices, label_map, coarse_labels

###### dataset functions ######

class CatFaceDataset:
    def __init__(self, partition, load_w=True, **kwargs):

        root = 'dataset/catface/images'
        self.load_w = load_w
        self.dset = ImageDataset(root, return_path=True, **kwargs)
        partition_idx, label_map, coarse_labels = get_partition_indices_pets(partition, self.dset)

        self.dset = Subset(self.dset, partition_idx)
        self.partition_idx = partition_idx
        self.coarse_labels = coarse_labels
        self.label_map = label_map

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data = self.dset[idx] # image, label, path
        # replace the label
        label_string = os.path.basename(data[2]).split('_')[0]
        label = self.label_map[label_string]
        if self.load_w:
            path = data[2]
            w_path = (path.replace('/images/images/', '/latents/')
                      .replace('png', 'pth'))
            # remove batch dim
            w_pth = torch.load(w_path, map_location='cpu')['w'][0].detach()
            return (data[0], w_pth, label, *data[2:])
        return (data[0], label, *data[2:])


###### transformation functions ######

def get_transform(dataset, transform_type):
    assert(dataset.lower() == 'cat' or dataset.lower() == 'catface')
    if transform_type == 'imtrain':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                         interpolation=Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'imval':
        return transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'im2tensor':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'imcrop':
        return transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif transform_type == 'tensorbase':
        return TensorBaseTransform()
    elif transform_type == 'tensormixed':
        return TensorCombinedTransform()
    elif transform_type == 'tensormixedtrain':
        return TensorCombinedTransformTrain()
    else:
        raise NotImplementedError

class TensorBaseTransform(object):
    def __init__(self):
        self.crop_size = 224

    def __call__(self, image):
        # crops 256x256 --> 224x224
        image = centercrop_tensor(image, self.crop_size, self.crop_size)
        return image

# mimics RandomCrop and RandomHorizontalFlip on Tensor inputs
class TensorCombinedTransform(object):
    def __init__(self):
        self.crop_size = 224

    def __call__(self, image):
        # applies a different random crop / random flip for each image
        # in the batch
        # random crop at 224 
        image = torch.cat([randomcrop_tensor(im[None], self.crop_size)
                           for im in image])
        # random horizontal flip 
        image = torch.stack([torch.flip(x, dims=(-1,)) if torch.rand(1) >
                             0.5 else x for x in image])
        return image

# mimics RandomResizeCrop and RandomHorizontalFlip on Tensor inputs
class TensorCombinedTransformTrain(object):
    def __init__(self):
        self.crop_size = 224
        self.resize = torch.nn.Upsample((self.crop_size, self.crop_size), mode='bilinear')
        self.scale = (0.8, 1.0)
        self.ratio = (3. / 4., 4. / 3.)
        self.flip = True

    def get_image_size(self, img):
        assert(isinstance(img, torch.Tensor) and img.dim() > 2)
        return img.shape[-2:][::-1]

    def get_params(self, img):
        scale = self.scale
        ratio = self.ratio
        width, height = self.get_image_size(img)
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, image):
        # random resized crop
        image_list = []
        for im in image:
            im = im[None]
            i, j, h, w = self.get_params(im)
            im = im[:, :, i:i+h, j:j+w] # random resize crop
            im = self.resize(im) # resize to output size
            image_list.append(im)
        image = torch.cat(image_list, dim=0)
        # horizontal flip
        if self.flip:
            image = torch.stack([torch.flip(x, dims=(-1,)) if torch.rand(1) >
                                 0.5 else x for x in image])
        return image

