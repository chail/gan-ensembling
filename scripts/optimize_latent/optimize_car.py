from __future__ import print_function
import argparse
import os
import torch
import math
import numpy as np
from PIL import Image
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']

import data
import data.transforms
from networks import domain_generator
from utils import util, renormalize

def optimize(opt):
    dataset_name = 'car'
    generator_name = 'stylegan2'

    transform = data.get_transform(dataset_name, 'im2tensor')

    # loads the PIL image
    dset = data.get_dataset(dataset_name, opt.partition,
                            load_w=False, transform=None)
    total = len(dset)

    if opt.indices is None:
        start_idx = 0
        end_idx = total
    else:
        start_idx = opt.indices[0]
        end_idx = opt.indices[1]

    print("Optimizing dataset partition %s items %d to %d" %
          (opt.partition, start_idx, end_idx))

    generator = domain_generator.define_generator(
        generator_name, dataset_name, load_encoder=True)
    util.set_requires_grad(False, generator.generator)
    util.set_requires_grad(False, generator.encoder)

    for i in range(start_idx, end_idx):
        (im, label, bbox, path) = dset[i]
        img_filename = os.path.splitext(os.path.basename(path))[0]

        print("Running %d / %d images: %s" %
              (i, end_idx, img_filename))

        output_filename = os.path.join(opt.w_path, img_filename)
        if os.path.isfile(output_filename + '.pth'):
            print(output_filename + '.pth found... skipping')
            continue

        # scale image to 512 width
        width, height = im.size
        ratio = 512 / width
        new_width = 512
        new_height = int(ratio*height)
        new_im = im.resize((new_width, new_height), Image.ANTIALIAS)
        print(im.size)
        print(new_im.size)
        bbox = [int(x * ratio) for x in bbox]

        # shift to center the bbox annotation
        cx = (bbox[2] + bbox[0]) // 2
        cy = (bbox[3] + bbox[1]) // 2
        print("%d --> %d" % (cx, new_width // 2))
        print("%d --> %d" % (cy, new_height // 2))
        offset_x = new_width // 2 - cx
        offset_y = new_height // 2 - cy

        im_tensor = transform(new_im)
        im_tensor, mask = data.transforms.shift_tensor(
            im_tensor, offset_y, offset_x)
        im_tensor = data.transforms.centercrop_tensor(im_tensor, 384, 512)
        mask = data.transforms.centercrop_tensor(mask, 384, 512)
        # now image size is at most 512 x 384 (could be smaller)

        # center car on 512x512 tensor
        disp_y = (512 - im_tensor.shape[1])//2
        disp_x = (512 - im_tensor.shape[2])//2
        centered_im = torch.ones((3, 512, 512)) * 0
        centered_im[:, disp_y:disp_y+im_tensor.shape[1], disp_x:disp_x+im_tensor.shape[2]] = im_tensor
        centered_mask = torch.zeros_like(centered_im)
        centered_mask[:, disp_y:disp_y+im_tensor.shape[1], disp_x:disp_x+im_tensor.shape[2]] = mask

        ckpt, loss = generator.optimize(centered_im[None].cuda(),
                                        centered_mask[:1][None].cuda())

        w_optimized = ckpt['current_z']
        loss = np.array(loss).squeeze()
        im_optimized = renormalize.as_image(ckpt['current_x'][0])
        torch.save({'w': w_optimized.detach().cpu()},
                   output_filename + '.pth')
        np.savez(output_filename + '_loss.npz', loss=loss)
        im_optimized.save(output_filename + '_optimized_im.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, required=True,
                        help='specify train, val, or test partition')
    parser.add_argument('--w_path', type=str, required=True,
                        help='directory to save the optimized latents')
    parser.add_argument('--indices', type=int, nargs=2,
                        help='optimize latents for specific image range')
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.w_path, exist_ok=True)
    optimize(opt)
