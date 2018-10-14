import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def ifgsm(model, X, y, niters=10, epsilon=0.01, learning_rate=0.01):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for i in range(niters):
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
        
        X_pert.volatile = False;
        
    return X_pert

def pgd(model, X, y, niters=10, epsilon=0.01, learning_rate=0.01):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for i in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        grad = X_pert.grad.detach()
        pert = learning_rate/torch.max(grad) * grad

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True
        
        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True
        
        X_pert.volatile = False
        
    return X_pert

def min_indices(x):
    y = torch.LongTensor(x.size(0)).zero_()
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            x_ = x.clone().cpu().data.numpy()
            if np.min(x_, 1)[i] == x_[i][j]:
                y[i] = j
                break
    return Variable(y).cuda()

def step_ll(model, X, y, niters=10, epsilon=0.01, learning_rate=0.01):
    X_pert = X.clone()
    X_pert.requires_grad = True
    output_normal = model(X_pert)
    output_min = min_indices(output_normal)
    
    for i in range(niters):
        X_pert.volatile = False
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, output_min)
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # add perturbation
        X_pert = X_pert.detach() - pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True
        
        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True
        
    return X_pert
