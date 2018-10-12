#to be run with torch-0.4
import torch
import torchvision
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import os.path
import argparse
import torch.nn as nn

from cifar10models import *
from torchvision.models import *
from torch.autograd import Variable


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

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--eps', type=float, default=0.01, help='epsilon value for ifgsm attack')
parser.add_argument('--niters', type=int, default=10, help='number of iterations for ifgsm attack')
parser.add_argument('--adv_rate', type=float, default=0.01, help='learning rate of adversarial examples')
parser.add_argument('--model', type=string, default="", help='googlenet, densenet121, resnet18, senet18')
parser.add_argument('--weights_path', type=string, default='../../../../share/cuvl/weights/cifar10/', help='weight directory')


def ifgsm(model, X, y, niters=10, epsilon=0.01, visualize=False):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = epsilon * X_pert.grad.detach().sign()

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

model = model_dict(args.model)
model.load_state_dict(torch.load(os.path.join(args.weights_path, weights_dict[args.model])))

def _data_transforms_cifar10():
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

train_tensors = []
train_label = []

for i in range(num_train):
    train_tensors.append(train_data[i][0])
    train_label.append(train_data[i][1])

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(0.5 * num_train))

train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=64,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
    pin_memory=True, num_workers=2)


perturbed_images = []
perturbed_labels = []
for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    
    input = ifgsm(model, input, target, eps=args.eps, niters=args.niters, learning_rate=args.adv_rate)
    perturbed_images.append(input)
    perturbed_labels.append(target)
    
pert_imgs = torch.cat(perturbed_images)
pert_lbls = torch.cat(perturbed_labels)
save_lbl = args.model + '_' + args.eps + '_' + args.niters + '_' + args.adv_rate
np.save(save_lbl + '_imgs.npy', pert_imgs.cpu().numpy())
np.save(save_lbl + '_lbls.npy', pert_lbls.cpu().numpy())