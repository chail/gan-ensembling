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

import data
from utils import util, netinit, pbar, pidfile
from networks.classifiers import attribute_classifier, attribute_utils

def train(opt):
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    device = 'cuda'
    batch_size = int(opt.batch_size)

    # tensorboard
    os.makedirs(os.path.join(opt.outf, 'runs'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.outf, 'runs'))


    # classifier follows architecture and initialization from attribute
    # classifiers in stylegan: 
    # https://github.com/NVlabs/stylegan/blob/master/metrics/linear_separability.py#L136
    # https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L564
    net = attribute_classifier.D(3, resolution=256, fixed_size=True, use_mbstd=False)
    # use random normal bc wscale will rescale weights, see tf source
    # https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L148
    netinit.init_weights(net, init_type='normal', gain=1.)
    net = net.to(device)

    # losses + optimizers
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # datasets -- added random horizonal flipping for training
    train_transform = data.get_transform('celebahq', 'imtrain')
    train_dset = data.get_dataset('celebahq', 'train', opt.attribute,
                                  load_w=False, transform=train_transform)
    print("Training transform:")
    print(train_transform)
    val_transform = data.get_transform('celebahq', 'imval')
    val_dset = data.get_dataset('celebahq', 'val', opt.attribute,
                                load_w=False, transform=val_transform)
    print("Validation transform:")
    print(val_transform)
    train_loader = DataLoader(train_dset, batch_size=opt.batch_size,
                              shuffle=True, pin_memory=False,
                              num_workers=opt.workers)
    val_loader = DataLoader(val_dset, batch_size=opt.batch_size,
                            shuffle=False, pin_memory=False,
                            num_workers=opt.workers)
    start_ep = 0
    best_val_acc = 0.0
    best_val_epoch = 0

    for epoch in pbar(range(start_ep, opt.niter+1)):
        # average meter for train/val loss and train/val acc
        metrics=dict(train_loss=util.AverageMeter(),
                     val_loss=util.AverageMeter(),
                     train_acc=util.AverageMeter(),
                     val_acc=util.AverageMeter())

        # train loop
        for step, (im, label) in enumerate(pbar(train_loader)):
            im = im.cuda()
            label = label.cuda().float()

            net.zero_grad()
            logit, softmaxed = attribute_utils.get_softmaxed(net, im)
            # enforces that negative logit --> our label = 1
            loss = bce_loss(logit, 1-label)
            predicted = (softmaxed > 0.5).long()
            correct = (predicted == label).float().mean().item()
            metrics['train_loss'].update(loss, n=len(label))
            metrics['train_acc'].update(correct, n=len(label))
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                pbar.print("%s: %0.2f" % ('train loss', metrics['train_loss'].avg))
                pbar.print("%s: %0.2f" % ('train acc', metrics['train_acc'].avg))

        # val loop
        net = net.eval()
        with torch.no_grad():
            for step, (im, label) in enumerate(pbar(val_loader)):
                im = im.cuda()
                label = label.cuda().float()
                logit, softmaxed = attribute_utils.get_softmaxed(net, im)
                predicted = (softmaxed > 0.5).long()
                correct = (predicted == label).float().mean().item()
                loss = bce_loss(logit, 1-label)
                metrics['val_loss'].update(loss, n=len(label))
                metrics['val_acc'].update(correct, n=len(label))
        net = net.train()

        # send losses to tensorboard
        for k,v in metrics.items():
            pbar.print("Metrics at end of epoch")
            pbar.print("%s: %0.4f" % (k, v.avg))
            writer.add_scalar(k.replace('_', '/'), v.avg, epoch)

        # do checkpoint as latest
        util.make_checkpoint(net, optimizer, epoch, metrics['val_acc'].avg, opt.outf, 'latest')

        if metrics['val_acc'].avg > best_val_acc:
            pbar.print("Updating best checkpoint at epoch %d" % epoch)
            pbar.print("Old Best Epoch %d Best Val %0.2f" %
                       (best_val_epoch, best_val_acc))
            # do checkpoint as best
            util.make_checkpoint(net, optimizer, epoch, metrics['val_acc'].avg, opt.outf, 'best')
            best_val_acc = metrics['val_acc'].avg
            best_val_epoch = epoch
            pbar.print("New Best Epoch %d Best Val %0.2f" %
                       (best_val_epoch, best_val_acc))
            with open("%s/best_val.txt" % opt.outf, "w") as f:
                f.write("Best Epoch %d Best Val %0.2f\n" % (best_val_epoch, best_val_acc))

        if epoch >= best_val_epoch + 5:
            pbar.print("Exiting training")
            pbar.print("Best Val epoch %d" % best_val_epoch)
            pbar.print("Curr epoch %d" % epoch)
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--outf', type=str, help='output directory')

    opt = parser.parse_args()
    print(opt)

    if opt.outf is None:
        opt.outf = 'results/classifiers/celebahq/%s' % opt.attribute

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    pidfile.exit_if_job_done(opt.outf)
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)
    train(opt)
    print("finished training!")
    pidfile.mark_job_done(opt.outf)
