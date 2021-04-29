import torch
import numpy as np
import os
from data.image_dataset import ImageDataset
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image
import random
import math

from mat4py import loadmat
from collections import defaultdict
from .transforms import randomcrop_tensor, centercrop_tensor


###### utility functions ######
anno_path = 'dataset/cars/devkit/cars_meta.mat'
data = loadmat(anno_path)
class_names = data['class_names']

anno_path = 'dataset/cars/devkit/cars_train_annos.mat'
data = loadmat(anno_path)
class_labels = data['annotations']['class']
file_names = data['annotations']['fname']
bbox_annotations = (data['annotations']['bbox_x1'],
                    data['annotations']['bbox_y1'],
                    data['annotations']['bbox_x2'],
                    data['annotations']['bbox_y2'])

# for each class, which images are a member of it
indices_per_class = defaultdict(list)
for i in range(len(class_labels)):
    indices_per_class[class_labels[i]].append(i)

train_indices = {k: v[:len(v)//2] for k, v in indices_per_class.items()}
val_indices = {k: v[len(v)//2:-len(v)//4] for k, v in indices_per_class.items()}
test_indices = {k: v[-len(v)//4:] for k, v in indices_per_class.items()}

car_types = ['-'] + [x.split(' ')[-2] for x in class_names] # - for 1-indexing

def get_partition_indices(part, valid_classes):
    fine_to_coarse_label_map = {i: valid_classes.index(x) for
                                i, x in enumerate(car_types) if x in
                                valid_classes}
    partition_indices = []
    fine_labels = []
    if part == 'train':
        [partition_indices.extend(train_indices[k]) for k in fine_to_coarse_label_map]
        [fine_labels.extend([k]*len(train_indices[k])) for k in fine_to_coarse_label_map]
    if part == 'val':
        [partition_indices.extend(val_indices[k]) for k in fine_to_coarse_label_map]
        [fine_labels.extend([k]*len(val_indices[k])) for k in fine_to_coarse_label_map]
    if part == 'test':
        [partition_indices.extend(test_indices[k]) for k in fine_to_coarse_label_map]
        [fine_labels.extend([k]*len(test_indices[k])) for k in fine_to_coarse_label_map]
    return partition_indices, fine_labels, fine_to_coarse_label_map

###### dataset functions ######

class CarsDataset:
    def __init__(self, partition, classes='threecars', load_w=True, **kwargs):
        root = 'dataset/cars/images'
        self.load_w = load_w
        self.dset = ImageDataset(root, return_path=True, **kwargs)

        if classes == 'sixcars':
            valid_classes = ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Cab']
        elif classes == 'threecars':
            valid_classes = ['SUV', 'Sedan', 'Cab']
        elif classes == 'suvsedan':
            valid_classes = ['SUV', 'Sedan']
        else:
            valid_classes = None
        print("The valid classes are: %s" % valid_classes)
        partition_idx, fine_labels, fine_to_coarse_label_map = get_partition_indices(partition, valid_classes)

        self.dset = Subset(self.dset, partition_idx)
        self.fine_to_coarse_label_map = fine_to_coarse_label_map
        self.partition_idx = partition_idx
        self.fine_labels = fine_labels
        self.coarse_labels = valid_classes

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data = self.dset[idx]
        fine_label = self.fine_labels[idx]
        img_filename = os.path.splitext(os.path.basename(data[2]))[0]
        coarse_label = self.fine_to_coarse_label_map[fine_label]

        remapped_index = self.dset.indices[idx] # remap since it is subset
        assert(os.path.splitext(file_names[remapped_index])[0] == img_filename)
        assert(class_labels[remapped_index] == fine_label)

        # get bounding box
        bbox = (bbox_annotations[0][remapped_index],
                bbox_annotations[1][remapped_index],
                bbox_annotations[2][remapped_index],
                bbox_annotations[3][remapped_index])

        if self.load_w:
            path = data[2]
            w_path = path.replace('images/images', 'latents').replace('jpg', 'pth')
            # remove batch dimension
            w_pth = torch.load(w_path, map_location='cpu')['w'][0].detach()

            return (data[0], w_pth, coarse_label, bbox, *data[2:])

        return (data[0], coarse_label, bbox, *data[2:])

###### transformation functions #####

def get_transform(dataset, transform_type):
    assert(dataset.lower() == 'car')
    if transform_type == 'imtrain':
        return transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0),
                                         interpolation=Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif transform_type == 'imval':
        return transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.CenterCrop(256),
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
            transforms.RandomCrop(256),
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
        self.load_size = (256, 342) # 1.5x = (384, 512) gan output size
        self.crop_size = 256
        self.resize = torch.nn.Upsample(self.load_size, mode='bilinear')

    def __call__(self, image):
        image = image[:, :, 64:-64, :] # crop off the black padding
        image = self.resize(image)
        image = centercrop_tensor(image, self.crop_size, self.crop_size)
        return image

# mimics RandomCrop and RandomHorizontalFlip on Tensor inputs
class TensorCombinedTransform(object):
    def __init__(self):
        self.load_size = (256, 342) # 1.5x (512, 384)
        self.crop_size = 256
        self.resize = torch.nn.Upsample(self.load_size, mode='bilinear')

    def __call__(self, image):
        image = image[:, :, 64:-64, :] # crop off the black padding
        # resize 512x384 --> (342x256) (about 1.5x scale)
        image = self.resize(image)
        # random crop at 256 (for each image individually)
        image = torch.cat([randomcrop_tensor(im[None], self.crop_size)
                           for im in image])
        # random horizontal flip 
        image = torch.stack([torch.flip(x, dims=(-1,)) if torch.rand(1) >
                             0.5 else x for x in image])
        return image


# mimics RandomResizeCrop and RandomHorizontalFlip on Tensor inputs
class TensorCombinedTransformTrain(object):
    def __init__(self):
        # performs a tensor random-resize crop
        self.crop_size = 256
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
        image = image[:, :, 64:-64, :] # crop off the black badding
        # random resized crop, for each image independently
        image_list = []
        for im in image:
            im = im[None]
            i, j, h, w = self.get_params(image)
            im = im[:, :, i:i+h, j:j+w] # random resize crop
            im = self.resize(im) # resize to output size
            image_list.append(im)
        image = torch.cat(image_list, dim=0)
        # horizontal flip for each image independently
        image = torch.stack([torch.flip(x, dims=(-1,)) if torch.rand(1) >
                             0.5 else x for x in image])
        return image

