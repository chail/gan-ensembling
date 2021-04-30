import argparse
import torch
import numpy as np
import os

import data
from networks import domain_generator, domain_classifier
from utils import util


def optimize(opt):
    dataset_name = 'cifar10'
    generator_name = 'stylegan2-cc'  # class conditional stylegan
    transform = data.get_transform(dataset_name, 'imval')

    dset = data.get_dataset(dataset_name, opt.partition,
                            load_w=False, transform=transform)
    total = len(dset)
    if opt.indices is None:
        start_idx = 0
        end_idx = total
    else:
        start_idx = opt.indices[0]
        end_idx = opt.indices[1]

    generator = domain_generator.define_generator(
        generator_name, dataset_name, load_encoder=False)
    util.set_requires_grad(False, generator.generator)

    resnet = domain_classifier.define_classifier(dataset_name,
                                                 'imageclassifier')

    ### iterate ###
    for i in range(start_idx, end_idx):
        img, label = dset[i]

        print("Running img %d/%d" % (i, len(dset)))
        filename = os.path.join(opt.w_path, '%s_%06d.npy' %
                                (opt.partition, i))
        if os.path.isfile(filename):
            print(filename + ' found... skipping')
            continue

        img = img[None].cuda()
        with torch.no_grad():
            pred_logit = resnet(img)
            _, pred_label = pred_logit.max(1)
            pred_label = pred_label.item()
        print("True label %d prd label %d" % (label, pred_label))
        ckpt, loss = generator.optimize(img, pred_label)
        current_z = ckpt['current_z'].detach().cpu().numpy()
        np.save(filename, current_z)


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
