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
from utils import util, pbar, pidfile
from utils.metrics import accuracy
import torchvision.models

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
    train_dset = data.get_dataset(domain, 'train', load_w=False,
                                  transform=train_transform)
    print("Training transform:")
    print(train_transform)
    val_transform = data.get_transform(domain, 'imval')
    val_dset = data.get_dataset(domain, 'val', load_w=False,
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
    if not opt.train_from_scratch:
        state_dict = torchvision.models.utils.load_state_dict_from_url(
            torchvision.models.resnet.model_urls['resnet18'])
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        net.load_state_dict(state_dict, strict=False)
    net = net.to(device)

    # losses + optimizers + scheduler
    # use smaller learning rate for the feature layers if initialized 
    # with imagenet pretrained weights
    criterion = nn.CrossEntropyLoss().to(device) # loss(output, target)
    fc_params = [k[1] for k in net.named_parameters() if
                 k[0].startswith('fc')]
    feat_params = [k[1] for k in net.named_parameters() if
                   not k[0].startswith('fc')]
    feature_backbone_lr = opt.lr if opt.train_from_scratch else 0.1*opt.lr
    print("Initial learning rate for feature backbone: %0.4f" %
          feature_backbone_lr)
    print("Initial learning rate for FC layer: %0.4f" % opt.lr)
    optimizer = optim.Adam([{'params': fc_params},
                            {'params': feat_params,
                             'lr': feature_backbone_lr}],
                            lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, min_lr=1e-6, verbose=True)

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
        for step, item in enumerate(pbar(train_loader)):
            im = item[0].cuda()
            label = item[1].cuda()

            net.zero_grad()
            output = net(im)
            loss = criterion(output, label)

            accs, _  = accuracy(output, label, topk=(1,))
            metrics['train_loss'].update(loss, n=len(label))
            metrics['train_acc'].update(accs[0], n=len(label))
            loss.backward()
            optimizer.step()
        pbar.print("%s: %0.2f" % ('train loss', metrics['train_loss'].avg))
        pbar.print("%s: %0.2f" % ('train acc', metrics['train_acc'].avg))


        # val loop
        net = net.eval()
        with torch.no_grad():
            for step, item in enumerate(pbar(val_loader)):
                im = item[0].cuda()
                label = item[1].cuda()
                output = net(im)
                loss = criterion(output, label)
                accs, _ = accuracy(output, label, topk=(1,))
                metrics['val_loss'].update(loss, n=len(label))
                metrics['val_acc'].update(accs[0], n=len(label))
        net = net.train()

        # update scheduler
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

        # terminate training if reached min LR and best validation is
        # not improving
        if (float(optimizer.param_groups[0]['lr']) <= 1e-6 and
            epoch >= best_val_epoch + 20):
            pbar.print("Exiting training")
            pbar.print("Best Val epoch %d" % best_val_epoch)
            pbar.print("Curr epoch %d" % epoch)
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True, choices=['cat', 'car'])
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for FC layer (lr for feature backbone is set 10x lower)')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--train_from_scratch', action='store_true', help='if specified, does not initialize using imagenet pretrained weights')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--outf', type=str, help='output directory')

    opt = parser.parse_args()
    print(opt)

    if opt.outf is None:
        opt.outf = 'results/classifiers/%s/imageclassifier' % opt.domain
        if opt.train_from_scratch:
            opt.outf += '_from_scratch'

    os.makedirs(opt.outf, exist_ok=True)
    # save options
    pidfile.exit_if_job_done(opt.outf)
    with open(os.path.join(opt.outf, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    train(opt)
    print("finished training!")
    pidfile.mark_job_done(opt.outf)
