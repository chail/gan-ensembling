from __future__ import print_function
import argparse
import os
import torch
import numpy as np
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']

import data
from networks import domain_generator
from utils import util, renormalize

def optimize(opt):
    dataset_name = 'cat'
    generator_name = 'stylegan2'

    transform = data.get_transform(dataset_name, 'im2tensor')
    dset = data.get_dataset(dataset_name, opt.partition, load_w=False, transform=transform)

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
        (im, label, path) = dset[i]
        img_filename = os.path.splitext(os.path.basename(path))[0]

        print("Running %d / %d images: %s" % (i, end_idx, img_filename))

        output_filename = os.path.join(opt.w_path, img_filename)
        if os.path.isfile(output_filename + '.pth'):
            print(output_filename + '.pth found... skipping')
            continue

        # cat face dataset is already centered
        centered_im = im[None].cuda()
        # find zero values to estimate the mask
        mask = torch.ones_like(centered_im)
        mask[torch.where(torch.sum(torch.abs(centered_im), axis=0, keepdims=True) < 0.02)] = 0
        mask = mask[:, :1, :, :].cuda()
        ckpt, loss = generator.optimize(centered_im, mask=mask)

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
