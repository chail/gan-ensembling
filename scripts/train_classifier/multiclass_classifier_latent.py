from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import oyaml as yaml
import numpy as np

import data
from utils import util, pbar, pidfile
from utils.metrics import accuracy
from networks import domain_generator
import torchvision.models
os.environ['TORCH_EXTENSIONS_DIR'] = '/tmp/%s_cpp/' % os.environ['USER']


def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    device = 'cuda'
    batch_size = int(opt.batch_size)
    domain = opt.domain

    # tensorboard
    os.makedirs(os.path.join(opt.outf, 'runs'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.outf, 'runs'))

    #  datasets
    train_transform = data.get_transform(domain, 'imtrain')
    train_dset = data.get_dataset(domain, 'train', load_w=True,
                                  transform=train_transform)
    print("Training transform:")
    print(train_transform)
    val_transform = data.get_transform(domain, 'imval')
    val_dset = data.get_dataset(domain, 'val', load_w=True,
                                transform=val_transform)
    print("Validation transform:")
    print(val_transform)
    train_loader = DataLoader(train_dset, batch_size=opt.batch_size,
                              shuffle=True, pin_memory=False,
                              num_workers=opt.workers)
    val_loader = DataLoader(val_dset, batch_size=opt.batch_size,
                            shuffle=False, pin_memory=False,
                            num_workers=opt.workers)

    # classifier: resnet18 model
    net = torchvision.models.resnet18(num_classes=len(
        train_dset.coarse_labels))
    # load the model weights to finetune from
    ckpt_path = ('results/classifiers/%s/%s/net_best.pth' %
                 (opt.domain, opt.finetune_from))
    print("Finetuning model from %s" % ckpt_path)
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    net.load_state_dict(state_dict)
    net = net.to(device)

    # losses + optimizers
    criterion = nn.CrossEntropyLoss().to(device) # loss(output, target)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, min_lr=1e-6, verbose=True)

    start_ep = 0
    best_val_acc = 0.0
    best_val_epoch = 0

    # val tensor transform does not flip, train tensor transform
    # mimics random resized crop and random flip
    val_tensor_transform = data.get_transform(domain, 'tensorbase')
    train_tensor_transform = data.get_transform(domain, 'tensormixedtrain')

    # load GAN
    generator = domain_generator.define_generator('stylegan2', domain)


    for epoch in pbar(range(start_ep, opt.niter+1)):
        # average meter for train/val loss and train/val acc
        metrics=dict(train_loss=util.AverageMeter(),
                     val_loss=util.AverageMeter(),
                     train_acc=util.AverageMeter(),
                     val_acc=util.AverageMeter())

        # train loop
        for step, item in enumerate(pbar(train_loader)):
            im_orig = item[0].cuda()
            opt_w = item[1].cuda()
            label = item[2].cuda()
            with torch.no_grad():
                if opt.perturb_type == 'stylemix':
                    seed = epoch * len(train_loader) + step
                    mix_latent = generator.seed2w(n=opt_w.shape[0], seed=seed)
                    generated_im = generator.perturb_stylemix(
                        opt_w, opt.perturb_layer, mix_latent,
                        n=opt_w.shape[0], is_eval=False)
                elif opt.perturb_type == 'isotropic':
                    # used minimum value for isotropic setting
                    eps = np.min(generator.perturb_settings['isotropic_eps_%s' % opt.perturb_layer])
                    generated_im = generator.perturb_isotropic(
                        opt_w, opt.perturb_layer, eps=eps,
                        n=opt_w.shape[0], is_eval=False)
                elif opt.perturb_type == 'pca':
                    # used median value for pca setting
                    eps = np.median(generator.perturb_settings['pca_eps'])
                    generated_im = generator.perturb_pca(
                        opt_w, opt.perturb_layer, eps=eps,
                        n=opt_w.shape[0], is_eval=False)
                else:
                    generated_im = generator.decode(opt_w)

                generated_im = train_tensor_transform(generated_im)
                # sanity check that the shapes match
                assert(generated_im.shape == im_orig.shape)

            # with 50% chance, take the original image for
            # training the classifier
            im = im_orig if torch.rand(1) > 0.5 else generated_im

            net.zero_grad()
            output = net(im)
            loss = criterion(output, label)
            accs, _  = accuracy(output, label, topk=(1,))
            metrics['train_loss'].update(loss, n=len(label))
            metrics['train_acc'].update(accs[0], n=len(label))
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                pbar.print("%s: %0.6f" % ('train loss', metrics['train_loss'].avg))
                pbar.print("%s: %0.6f" % ('train acc', metrics['train_acc'].avg))

        # val loop
        net = net.eval()
        with torch.no_grad():
            for step, item in enumerate(pbar(val_loader)):
                im_orig = item[0].cuda()
                opt_w = item[1].cuda()
                label = item[2].cuda()
                with torch.no_grad():
                    # evaluate on the generated image
                    im = generator.decode(opt_w)
                    im = val_tensor_transform(im)
                assert(im.shape == im_orig.shape)
                output = net(im)
                loss = criterion(output, label)
                accs, _ = accuracy(output, label, topk=(1,))
                metrics['val_loss'].update(loss, n=len(label))
                metrics['val_acc'].update(accs[0], n=len(label))
        net = net.train()

        scheduler.step(metrics['val_acc'].avg)

        # send losses to tensorboard
        for k,v in metrics.items():
            pbar.print("Metrics at end of epoch")
            pbar.print("%s: %0.4f" % (k, v.avg))
            writer.add_scalar(k.replace('_', '/'), v.avg, epoch)
        pbar.print("Learning rate: %0.6f" % optimizer.param_groups[0]['lr'])

        # do checkpoint as latest
        util.make_checkpoint(net, optimizer, epoch, metrics['val_acc'].avg, opt.outf, 'latest')

        if metrics['val_acc'].avg > best_val_acc:
            pbar.print("Updating best checkpoint at epoch %d" % epoch)
            pbar.print("Old Best Epoch %d Best Val %0.6f" %
                       (best_val_epoch, best_val_acc))
            # do checkpoint as best
            util.make_checkpoint(net, optimizer, epoch, metrics['val_acc'].avg, opt.outf, 'best')
            best_val_acc = metrics['val_acc'].avg
            best_val_epoch = epoch
            pbar.print("New Best Epoch %d Best Val %0.6f" %
                       (best_val_epoch, best_val_acc))
            with open("%s/best_val.txt" % opt.outf, "w") as f:
                f.write("Best Epoch %d Best Val %0.6f\n" % (best_val_epoch, best_val_acc))

        if (float(optimizer.param_groups[0]['lr']) <= 1e-6 and
            epoch >= best_val_epoch + 50):
            pbar.print("Exiting training")
            pbar.print("Best Val epoch %d" % best_val_epoch)
            pbar.print("Curr epoch %d" % epoch)
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True, choices=['cat', 'car'])
    parser.add_argument('--finetune_from', type=str, default='imageclassifier', help='which model to finetune from')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')

    parser.add_argument('--perturb_type', type=str, help="which type of perturbation"
                        " to apply during training: isotropic, pca, stylemix",
                        choices=['isotropic', 'pca', 'stylemix'])
    parser.add_argument('--perturb_layer', type=str, help="which layer to apply"
                        " perturbation during training: fine, coarse",
                        choices=['fine', 'coarse'])
    parser.add_argument('--outf', type=str, help='output directory')

    opt = parser.parse_args()
    print(opt)

    if opt.outf is None:
        opt.outf = 'results/classifiers/%s/latentclassifier' % opt.domain
        if opt.perturb_type is not None:
            assert(opt.perturb_layer is not None)
            opt.outf += '_%s' % opt.perturb_type
            opt.outf += '_%s' % opt.perturb_layer

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    pidfile.exit_if_job_done(opt.outf)
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    train(opt)
    print("finished training!")
    pidfile.mark_job_done(opt.outf)
