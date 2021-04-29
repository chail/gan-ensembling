# modified from https://github.com/kuangliu/pytorch-cifar
'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import data
from networks import domain_generator
from networks.classifiers import cifar10_resnet
from networks.classifiers.cifar10_utils import progress_bar
from utils import pidfile


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# for finetuning, reduced the initial learning rate from 0.1 to 0.001
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--stylemix_layer', type=str, 
                    help='which layer to perform stylemixing (e.g. fine)')
args = parser.parse_args()

# setup output directory
if args.stylemix_layer is None:
    save_dir = 'results/classifiers/cifar10/latentclassifier'
else:
    save_dir = 'results/classifiers/cifar10/latentclassifier_stylemix_%s' % args.stylemix_layer
os.makedirs(save_dir, exist_ok=True)
pidfile.exit_if_job_done(save_dir)
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = data.get_transform('cifar10', 'imtrain')
transform_test = data.get_transform('cifar10', 'imval')
trainset = data.get_dataset('cifar10', 'train', load_w=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
# using validation partition
testset = data.get_dataset('cifar10', 'val', load_w=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

tensor_transform_train = data.get_transform('cifar10', 'tensortrain')
tensor_transform_val = data.get_transform('cifar10', 'tensorbase')

# Model
print('==> Building model..')
net = cifar10_resnet.ResNet18()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

# finetune it from the previous classifier
net.load_state_dict(torch.load('results/classifiers/cifar10/imageclassifier/ckpt.pth')['net'])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('%s/ckpt.pth' % save_dir)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200)

# load generator
generator = domain_generator.define_generator('stylegan2-cc', 'cifar10')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, ws, targets) in enumerate(trainloader):
        inputs, ws, targets = inputs.to(device), ws.to(device), targets.to(device)
        # with 50% prob, use inputs
        if torch.rand(1) > 0.5:
            images = inputs
        else:
            # otherwise use the specified GAN input (either just w, or 
            # mixed at a specified layer
            if args.stylemix_layer is not None:
                with torch.no_grad():
                    seed = epoch * len(trainloader) + batch_idx
                    # use GT labels for training
                    labels = (torch.nn.functional.one_hot(
                        targets, num_classes=generator.generator.c_dim)
                        .float().to(device))
                    mix_latent = generator.seed2w(
                        n=inputs.shape[0], seed=seed, labels=labels)
                    images = generator.perturb_stylemix(
                        ws, args.stylemix_layer, mix_latent,
                        n=inputs.shape[0], is_eval=False)
                    images = tensor_transform_train(images)
            else:
                # use the latents directly
                with torch.no_grad():
                    images = generator.decode(ws)
                    images = tensor_transform_train(images)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, ws, targets) in enumerate(testloader):
            inputs, ws, targets = inputs.to(device), ws.to(device), targets.to(device)
            # no stylemixing at test time 
            with torch.no_grad():
                images = generator.decode(ws)
                images = tensor_transform_val(images)
            outputs = net(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, '%s/ckpt.pth' % save_dir)
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

pidfile.mark_job_done(save_dir)
