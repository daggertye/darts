# This is an example of how to load a model, generate adversarial examples for it,
# and evaluate the transferability of these adversarial examples to other models

# Example output of this code:
# densenet121_epoch_7_acc_86.05.pth | Fooling Ratio: 82.5 , Perturbed Accuracy:  0.0 , Original Accuracy: 75.0
# ----------------------------------------------------------------------------------------------------
# resnet18_epoch_28_acc_89.08.pth | Fooling Ratio: 55.0 , Perturbed Accuracy:  37.5 , Original Accuracy: 80.0
# googlenet_epoch_8_acc_86.02.pth | Fooling Ratio: 75.0 , Perturbed Accuracy:  17.5 , Original Accuracy: 80.0

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

#from cifar10models import ResNet18, DenseNet121, GoogLeNet, SENet18
from attacks import ifgsm
from evaluation import test_adv_examples_across_models, test_adv_examples_across_full_models

# load and initialize the dataset
print('initializing...')
mean_arr, stddev_arr = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_arr,stddev_arr)])
testset = torchvision.datasets.CIFAR10(root='/share/cuvl/pytorch_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

test_tuples = tuple([tup for i, tup in enumerate(testloader) if i < 20])
test_ims, test_lbls = zip(*test_tuples)
test_ims, test_lbls = torch.cat(test_ims, 0), torch.cat(test_lbls, 0)
test_ims, test_lbls = test_ims.cuda(), test_lbls.cuda()
print('initializion complete!')


print('loading models...')

# define path and load models
PATH_4 = './search_full/'
#PATH_8 = './results/Layer_8/'

def load_model(path, n):
    new_model = torch.load(os.path.join(path,'weights_' + str(n) + '.pt'))
    return new_model

model_pair = []

# pick out some models that we need and put them in a dictionary.
for i in range(50):
    model_pair.append((load_model(PATH_4, i), 'model' + str(i)))

print('model loading complete!')


def adversarial_workbench(network_list=model_pair):
    fool_ratio_dict = []
    pert_acc_dict = []
    ori_acc_dict = []
    for i in range(0, len(model_pair)):
        base_model, base_model_name = model_pair[i]
        net = base_model.cuda()
        net.eval()
        
        adv_examples = ifgsm(net, test_ims, test_lbls, niters=10, epsilon=0.03)
        transfer_results = test_adv_examples_across_full_models(adv_examples,
                                               test_ims,
                                               test_lbls,
                                               network_list=model_pair,
                                              )
       
        fool_ratio_dict.append([])
        pert_acc_dict.append([])
        ori_acc_dict.append([])
        for j in range(0, len(transfer_results)):
            fool_ratio_dict[i].append(transfer_results[j][1])
            pert_acc_dict[i].append(transfer_results[j][2])
            ori_acc_dict[i].append(transfer_results[j][3])
     
    return fool_ratio_dict, pert_acc_dict, ori_acc_dict
        
fool_ratio, pert_acc, ori_acc = adversarial_workbench(model_pair)
print('Fooling ratio is: ')
print(np.matrix(fool_ratio))
print('Perturbation accuracy is: ')
print(np.matrix(pert_acc))
    
"""
print('crafting adversarial examples...')
model_4_10 = load_model(PATH_4, 10)
net = model_4_10.cuda()
net.eval()

# if you use a large number of test_ims, will likely have to batch calls to ifgsm,
# like in the example notebook
adv_examples = ifgsm(net, test_ims, test_lbls, niters=10, epsilon=0.03)
print('crafting adversarial examples completed!')

print('evaluating transferability...')
transfer_results = test_adv_examples_across_full_models(adv_examples,
                                               test_ims,
                                               test_lbls,
                                               network_list=model_pair,
                                              )
print('evaluating transferability complete!')


print('Number of images evaluating:', adv_examples.size(0))
densenet = transfer_results[0]
print(densenet[0].split('/')[-1], '| Fooling Ratio:', densenet[1], ', Perturbed Accuracy: ', densenet[2], ', Original Accuracy:', densenet[3])
print('-' * 100)

for i in range(1, len(transfer_results)):
    print(transfer_results[i][0].split('/')[-1], '| Fooling Ratio:', transfer_results[i][1], ', Perturbed Accuracy: ', transfer_results[i][2], ', Original Accuracy:', transfer_results[i][3])
"""
