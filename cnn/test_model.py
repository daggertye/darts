import torch
import torchvision
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import os.path
import argparse
import torch.nn as nn
import utils
import logging
import sys

from cifar10models import *
from torchvision.models import *
from torch.autograd import Variable
from tqdm import tqdm


model_dict = {
    'googlenet' : GoogLeNet,
    'densenet121' : DenseNet121,
    'resnet18' : ResNet18,
    'senet18' : SENet18
}

weights_dict = {
    'googlenet' : 'googlenet_epoch_227_acc_94.86.pth',
    'densenet121' : 'densenet121_epoch_315_acc_95.61.pth',
    'resnet18' : 'resnet18_epoch_347_acc_94.77.pth',
    'senet18' : 'senet18_epoch_279_acc_94.59.pth'
}
parser = argparse.ArgumentParser("cifar testing")
parser.add_argument('--eps', type=float, default=0.01, help='epsilon value for ifgsm attack')
parser.add_argument('--niters', type=int, default=10, help='number of iterations for ifgsm attack')
parser.add_argument('--adv_rate', type=float, default=0.01, help='learning rate of adversarial examples')
parser.add_argument('--model', type=str, default="", help='googlenet, densenet121, resnet18, senet18')
parser.add_argument('--weights_path', type=str, default='../../../../share/cuvl/weights/cifar10/', help='weight directory')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('attacked_results', args.model + '_' + str(args.eps) + '_' + str(args.niters) + '_' + str(args.adv_rate) + '.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def ifgsm(model, X, y, niters=10, epsilon=0.01, learning_rate=0.01):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True
        
        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True
        
    return X_pert

model = model_dict[args.model]()
model.cuda()
model.load_state_dict(torch.load(os.path.join(args.weights_path, weights_dict[args.model])))

def _data_transforms_cifar10():
    #CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    #CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

train_transform, valid_transform = _data_transforms_cifar10()
train_data = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(0.5 * num_train))

train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=64,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
    pin_memory=True, num_workers=2)

top1 = utils.AvgrageMeter()
top5 = utils.AvgrageMeter()

for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    
    input_pert = ifgsm(model, input, target, epsilon=args.eps, niters=args.niters, learning_rate=args.adv_rate)
    input_pert = input_pert.detach()
    logits = model(input_pert)
    
    n = input.size(0)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    
    if step % 50 == 0:
        logging.info('valid %03d %f %f', step, top1.avg, top5.avg)
    