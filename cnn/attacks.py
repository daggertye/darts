import torch
import torchvision
import torch.nn as nn

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
        pert = learning_rate/torch.max() * grad

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

def min_one_hot(x):
    y = torch.FloatTensor(x.size(0), x.size(1)).zero_()
    for i in range(len(x)):
        for j in range(x.size(1)):
            if x[i][j] == torch.min(x[i]):
                y[i][j] = 1
                break
    return y

def step_ll(model, X, y, niters=10, epsilon=0.01, learning_rate=0.01):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for i in range(niters):
        output_perturbed = model(X_pert)
        output_ll = min_one_hot(output_perturbed)
        loss = nn.CrossEntropyLoss()(output_ll, y)
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
        
        X_pert.volatile = False;
        
    return X_pert
