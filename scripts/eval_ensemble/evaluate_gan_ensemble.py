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
from networks import domain_classifier, domain_generator

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

# argument parsing
parser = argparse.ArgumentParser(description='Evaluate ensemble of GAN-generated augmentations.')
parser.add_argument('--domain', type=str, required=True,
                    help='which domain to evaluate: celebahq,cat,car')
parser.add_argument('--classifier_name', type=str, required=True,
                    help='which classifier to use')
parser.add_argument('--aug_type', type=str, required=True,
                    help='which type of GAN augmentation to use',
                    choices=['isotropic', 'pca', 'stylemix'])
parser.add_argument('--aug_layer', type=str, default='fine',
                    help='which layer to apply perturbation',
                    choices=['fine', 'coarse'])
parser.add_argument('--apply_tensor_transform', action='store_true',
                    help='applies a tensor transform (e.g. random crop) on GAN-generated output')
parser.add_argument('--generator_type', type=str, default='stylegan2',
                    help='which generator type to use. default stylegan2')
parser.add_argument('--partition', type=str, default='val')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_ens', type=int, default=32)
parser.add_argument('--seed', type=int, default=2)
args = parser.parse_args()
print(args)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# sanity check generator type
if args.domain == 'cifar10' and args.generator_type != 'stylegan2-cc':
    print("A non-class-conditional generator was specified for cifar10")
    print("Currently supports --generator_type stylegan2-cc for cifar10")
    exit(0)

# lock the experiment directory
lockdir = f'results/evaluations/{args.domain}/lockfiles/{args.classifier_name}_{args.partition}/gan_ensemble_{args.aug_type}_{args.aug_layer}'
if args.apply_tensor_transform:
    lockdir += '_tensortransform'
os.makedirs(lockdir, exist_ok=True)
pidfile.exit_if_job_done(lockdir, redo=False)

# data output filename
data_filename = lockdir.replace('lockfiles', 'output') + '.npz'
os.makedirs(os.path.dirname(data_filename), exist_ok=True)
print("saving result in: %s" % data_filename)

# load dataset and classifier
val_transform = data.get_transform(args.domain, 'imval')
if 'celebahq' in args.domain:
    # for celebahq, load the attribute-specific dataset
    attribute = args.classifier_name.split('__')[0]
    dset = data.get_dataset(args.domain, args.partition, attribute,
                            load_w=True, transform=val_transform)
else:
    dset = data.get_dataset(args.domain, args.partition, load_w=True,
                            transform=val_transform)
loader = DataLoader(dset, batch_size=1, shuffle=False,
                    pin_memory=False, num_workers=4)
classifier = domain_classifier.define_classifier(args.domain,
                                                 args.classifier_name)
if hasattr(dset, 'coarse_labels'):
    # sanity check that number of classifier labels matches
    # number of labels in dataset, for cat/car and resnet18 models
    assert(len(dset.coarse_labels) == classifier.fc.bias.shape[0])

# load the domain generator
generator = domain_generator.define_generator(args.generator_type,
                                              args.domain)


n_ens = args.n_ens
batch_size = args.batch_size
assert(n_ens % batch_size == 0) # sanity check: check it divides evenly

# set up output data structure
ensemble_data = {
    'original': [], # predictions of the original image
    'reconstructed': [], # predictions of the gan reconstruction
    'label': [], # GT label
    'random_seed': args.seed,
}
if args.aug_type == 'isotropic':
    eps_values = generator.perturb_settings['isotropic_eps_%s' % args.aug_layer]
    ensemble_expt_keys = ['isotropic_%s_%0.2f' % (args.aug_layer, eps)
                          for eps in eps_values]
    for expt_key in ensemble_expt_keys:
        ensemble_data[expt_key] = []
    print("Isotropic %s perturb eps: %s" % (args.aug_layer, eps_values))
elif args.aug_type == 'pca':
    eps_values = generator.perturb_settings['pca_eps']
    ensemble_expt_keys = ['pca_%s_%0.2f' % (args.aug_layer, eps)
                          for eps in eps_values]
    for expt_key in ensemble_expt_keys:
        ensemble_data[expt_key] = []
    print("PCA %s perturb eps: %s" % (args.aug_layer, eps_values))
elif args.aug_type == 'stylemix':
    ensemble_data['stylemix_' + args.aug_layer] = []

# define tensor transforms
# applies the standard validation center crop on GAN output
tensor_transform_val = data.get_transform(args.domain, 'tensorbase')
if args.apply_tensor_transform:
    # applies random crops on gan output if specified
    tensor_transform_ensemble = data.get_transform(args.domain, 'tensormixed')

for i, imdata_from_loader in enumerate(tqdm(loader)):
    # fix for too many open files error
    # https://github.com/pytorch/pytorch/issues/11201
    imdata = copy.deepcopy(imdata_from_loader)
    del imdata_from_loader
    im = imdata[0].cuda()
    opt_w = imdata[1].cuda()
    label = imdata[2]

    with torch.no_grad():
        ensemble_data['label'].append(label.numpy())

        # prediction on the image
        pred_original = domain_classifier.postprocess(classifier(im))
        ensemble_data['original'].append(pred_original.cpu().numpy())

        # prediction on the gan-reconstructed image
        reconstruction = generator.decode(opt_w.cuda())
        pred_rec = domain_classifier.postprocess(classifier(
            tensor_transform_val(reconstruction)))
        ensemble_data['reconstructed'].append(pred_rec.cpu().numpy())

        # handle isotropic or pca type gan perturb
        if args.aug_type == 'isotropic' or args.aug_type == 'pca':
            if args.aug_type == 'isotropic':
                perturb_fn = generator.perturb_isotropic
            elif args.aug_type == 'pca':
                perturb_fn = generator.perturb_pca
            for eps, expt_key in zip(eps_values, ensemble_expt_keys):
                perturbed_pred = []
                # if args.apply_tensor_transform is specified, it runs one
                # batch of GAN perturbations, and obtains additional
                # variations by applying the tensor_transform_ensemble to
                # each perturbed gan image to obtain the desired number
                # of variations (runs faster). Otherwise, if not specified
                # it obtains the full number of ensemble elements by 
                # perturbing the original image the desired number of 
                # times.
                total_gan_samples = batch_size if args.apply_tensor_transform else n_ens
                for batch_start in range(0, total_gan_samples, batch_size):
                    curr_batch_size = (min(batch_size+batch_start, n_ens)
                                       - batch_start)
                    perturbed_im = perturb_fn(opt_w, args.aug_layer,
                                              eps=eps,n=curr_batch_size)
                    if not args.apply_tensor_transform:
                        postprocessed_im = tensor_transform_val(perturbed_im)
                    else:
                        postprocessed_im = torch.cat([
                            tensor_transform_ensemble(perturbed_im)
                            for _ in range(n_ens // batch_size)
                        ], dim=0)
                    perturbed_pred.append(domain_classifier.postprocess(
                        classifier(postprocessed_im)).cpu().numpy())
                # sanity check: check that the desired number
                # of ensemble elements is reached
                perturbed_pred = np.concatenate(perturbed_pred)
                assert(len(perturbed_pred) == n_ens)
                ensemble_data[expt_key].append(perturbed_pred)
        # handle stylemix type gan perturb
        elif args.aug_type == 'stylemix':
            perturbed_pred = []
            total_gan_samples = batch_size if args.apply_tensor_transform else n_ens
            if args.generator_type=='stylegan2-cc':
                # use predicted labels to generated class-conditional
                # random samples
                lab = torch.zeros([total_gan_samples,
                                   generator.generator.c_dim],
                                  device=generator.device)
                # get the label from the original image prediction
                _, pred_label = pred_original.max(1)
                pred_label = pred_label.item()
                lab[:, pred_label] = 1
                mix_latent = generator.seed2w(seed=i, n=total_gan_samples,
                                              labels=lab)
            else:
                mix_latent = generator.seed2w(seed=i, n=total_gan_samples)
            for batch_start in range(0, total_gan_samples, batch_size):
                curr_batch_size = (min(batch_size+batch_start, n_ens)
                                   - batch_start)
                mix_latent_batch = mix_latent[batch_start:batch_start+curr_batch_size]
                perturbed_im = generator.perturb_stylemix(
                    opt_w, args.aug_layer, mix_latent_batch, n=curr_batch_size)
                if not args.apply_tensor_transform:
                    postprocessed_im = tensor_transform_val(perturbed_im)
                else:
                    postprocessed_im = torch.cat([
                        tensor_transform_ensemble(perturbed_im)
                        for _ in range(n_ens // batch_size)
                    ], dim=0)
                perturbed_pred.append(domain_classifier.postprocess(
                    classifier(postprocessed_im)).cpu().numpy())
            # sanity check: check that the desired number
            # of ensemble elements is reached
            perturbed_pred = np.concatenate(perturbed_pred)
            assert(len(perturbed_pred) == n_ens)
            ensemble_data['stylemix_' + args.aug_layer].append(perturbed_pred)

# save the result and unlock directory
np.savez(data_filename, **ensemble_data)
pidfile.mark_job_done(lockdir)
