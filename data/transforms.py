import torchvision.transforms as transforms
from utils import renormalize
import torch
import random
import numpy as np

def zc2imagenet_tensor(x):
    return renormalize.as_tensor(x, source='zc', target='imagenet')

def centercrop_tensor(x, crop_y, crop_x=None):
    if crop_x is None:
        crop_x = crop_y
    offset_y = max(0, (x.shape[-2] - crop_y) // 2)
    offset_x = max(0, (x.shape[-1] - crop_x) // 2)
    return x[..., offset_y:offset_y+crop_y, offset_x:offset_x+crop_x]

def randomcrop_tensor(x, crop):
    n, c, h, w = x.shape
    xp = random.randint(0, np.maximum(0, w - crop))
    yp = random.randint(0, np.maximum(0, h - crop))
    return x[..., yp:yp+crop, xp:xp+crop]

def shift_tensor(tensor, disp_y, disp_x):
    shape = tensor.shape
    y, x = shape[-2:]
    shifted_tensor = torch.ones_like(tensor) * -1
    mask = torch.zeros_like(tensor)

    extract_start_y = max(0, -disp_y)
    extract_start_x = max(0, -disp_x)
    extract_end_y = min(y, y - disp_y)
    extract_end_x = min(x, x - disp_x)
    extract_tensor = tensor[..., extract_start_y:extract_end_y,
                        extract_start_x:extract_end_x]
    paste_start_y = max(0, disp_y)
    paste_start_x = max(0, disp_x)
    paste_end_y = min(y, y+disp_y)
    paste_end_x = min(x, x+disp_x)
    shifted_tensor[..., paste_start_y:paste_end_y,
              paste_start_x:paste_end_x] = extract_tensor
    mask[..., paste_start_y:paste_end_y,
              paste_start_x:paste_end_x] = 1
    return shifted_tensor, mask

# ensemble the cropped images
class ImageEnsemble(object):
    def __init__(self, val_transform, ensemble_transform, n_ens):
        self.standard_transform = val_transform
        self.jitter_transform = ensemble_transform
        self.n_ens = n_ens

    def __call__(self, image):
        samples = ([self.standard_transform(image)] +
                   [self.jitter_transform(image) for _ in range(self.n_ens-1)])
        return torch.stack(samples)

