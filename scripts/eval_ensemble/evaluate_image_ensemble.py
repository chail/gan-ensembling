import sys,os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import pidfile
import random
import copy
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']

import data
from data.transforms import ImageEnsemble
from networks import domain_classifier

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

# argument parsing
parser = argparse.ArgumentParser(description='Evaluate ensemble of image augmentations.')
parser.add_argument('--domain', type=str, required=True,
                    help='which domain to evaluate: celebahq,cat,car')
parser.add_argument('--classifier_name', type=str, required=True,
                    help='which classifier to use')
parser.add_argument('--aug_type', type=str, default='imcrop',
                    help='which image transform to use for ensembling')
parser.add_argument('--partition', type=str, default='val')
parser.add_argument('--n_ens', type=int, default=32)
parser.add_argument('--seed', type=int, default=2)
args = parser.parse_args()
print(args)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# lock the experiment directory
lockdir = f'results/evaluations/{args.domain}/lockfiles/{args.classifier_name}_{args.partition}/image_ensemble_{args.aug_type}'
os.makedirs(lockdir, exist_ok=True)
pidfile.exit_if_job_done(lockdir, redo=False)

# data output filename
data_filename = lockdir.replace('lockfiles', 'output') + '.npz'
os.makedirs(os.path.dirname(data_filename), exist_ok=True)
print("saving result in: %s" % data_filename)

# load dataset and classifier
val_transform = data.get_transform(args.domain, 'imval')
ensemble_transform = data.get_transform(args.domain, args.aug_type)
transform = ImageEnsemble(val_transform, ensemble_transform, args.n_ens)
print("Ensemble transform:")
print(ensemble_transform)
if 'celebahq' in args.domain:
    # for celebahq, load the attribute-specific dataset
    attribute = args.classifier_name.split('__')[0]
    dset = data.get_dataset(args.domain, args.partition, attribute,
                            load_w=False, transform=transform)
else:
    dset = data.get_dataset(args.domain, args.partition, load_w=False,
                            transform=transform)
loader = DataLoader(dset, batch_size=1, shuffle=False,
                    pin_memory=False, num_workers=4)
classifier = domain_classifier.define_classifier(args.domain,
                                                 args.classifier_name)


# output data structure
ensemble_data = {
    'original': [], # predictions of the original image
    'label': [], # GT label
    args.aug_type: [], # ensembled predictions using specified image aug
    'random_seed': args.seed,
}

# do evaluation
for i, imdata_from_loader in enumerate(tqdm(loader)):
    # fix for too many open files error
    # https://github.com/pytorch/pytorch/issues/11201
    imdata = copy.deepcopy(imdata_from_loader)
    del imdata_from_loader
    im = imdata[0] # shape: 1xn_ensxCxHxW
    label = imdata[1]
    with torch.no_grad():
        ensemble_data['label'].append(label.numpy())
        im = im[0].cuda() # removes unused batch dim, keeps ensemble dim
        predictions = domain_classifier.postprocess(classifier(im))
        predictions_np = predictions.cpu().numpy()
        ensemble_data['original'].append(predictions_np[[0]])
        ensemble_data[args.aug_type].append(predictions_np[1:])

# save the result and unlock directory
np.savez(data_filename, **ensemble_data)
pidfile.mark_job_done(lockdir)
