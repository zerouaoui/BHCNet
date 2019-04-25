import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import mkdir_p
from util.misc import CSVLogger
from util.cutout import Cutout
from util.ERF import ERF

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']
scheduler_options = ['step', 'erf', 'cos', 'exp']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--dataset', default='cifar10', choices=dataset_options)
parser.add_argument('--model', default='resnet18', choices=model_options)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 20)')
parser.add_argument('--scheduler', default='step', choices=scheduler_options)
parser.add_argument('--rate', type=float, default=0.1, help='step decay rate for step scheduler')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False, help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, help='length of the holes')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
parser.add_argument('--alpha', type=int, default=-3, help='alpha for erf')
parser.add_argument('--beta', type=int, default=3, help='beta for erf')
parser.add_argument('--lr_min', type=float, default=0.0001, help='min learning rate')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)
if args.scheduler == 'step':
    basic_path = 'logs/' + args.dataset + '_' + args.model + '/rate_' + str(args.rate) + '/'
else:
    basic_path = 'logs/' + args.dataset + '_' + args.model + '/' + args.scheduler + '/'
mkdir_p(basic_path)
index = str(len(os.listdir(basic_path)) + 1)
basic_path = basic_path + '/' + index + '/'
mkdir_p(basic_path)
mkdir_p(basic_path + 'checkpoints/')

if args.scheduler == 'step':
    # E1/E1=rate, E1+E2=epochs
    args.E2 = args.epochs // (1 + args.rate)
    args.E1 = int(args.rate * args.E2)

csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc', 'max_acc'],
                       filename=basic_path + 'logs.csv')

# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root='data/',
                                  split='extra',
                                  transform=train_transform,
                                  download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.scheduler == 'step':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[args.E1, args.E2], gamma=0.1)
elif args.scheduler == 'erf':
    scheduler = ERF(cnn_optimizer, min_lr=args.lr_min, alpha=args.alpha, beta=args.beta, epochs=args.epochs)
elif args.scheduler == 'cos':
    scheduler = CosineAnnealingLR(cnn_optimizer, T_max=args.epochs)
elif args.scheduler == 'exp':
    scheduler = ExponentialLR(cnn_optimizer, gamma=0.98)


def eval(cnn, loader):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


max_acc = 0.

for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = eval(cnn, test_loader)
    if test_acc > max_acc:
        max_acc = test_acc
        torch.save(cnn.state_dict(), basic_path + 'checkpoints/max_acc.pt')

    scheduler.step(epoch)

    tqdm.write('test_acc: %.4f, max_acc: %.4f' % (test_acc, max_acc))
    csv_logger.writerow({
        'epoch': str(epoch),
        'train_acc': str(accuracy),
        'test_acc': str(test_acc),
        'max_acc': str(max_acc)
    })

torch.save(cnn.state_dict(), basic_path + 'checkpoints/last.pt')
csv_logger.close()
