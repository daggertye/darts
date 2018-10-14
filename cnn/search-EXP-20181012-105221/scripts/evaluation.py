import torch
from cifar10models import ResNet18, DenseNet121, GoogLeNet, SENet18

# standard testing given a network and dataloder, returns accuracy
def test_with_dataloader(network, dataloader, print_every=1000):
    network.eval()
    correct = 0
    total = 0
    for i, data in enumerate(dataloader, 0):
        if i > 1 and i % 1000 == 0:
            print('Evaluated', i, 'batches')
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100.0 * float(correct)/total

def test_checkpoint_with_dataloader(network, checkpoint, dataloader):
    net = network()
    net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    test_with_dataloader(net, dataloader)


# evaluates a set of adversarial examples against a single model, reporting relevant metrics
# input: adv_examples: torch.tensor with size Nx3x32x32
#        gt_labels: torch.tensor with size N
#        originals: torch.tensor with size Nx3x32x32
#        net: pytorch network to evaluate
# returns:
# (fooling_ratio, accuracy_perturbed, accuracy_original)
def test_adv_examples_for_model(adv_examples,
                                originals,
                                gt_labels,
                                net,
                                batch_size=4):
    net.eval()

    total = adv_examples.size(0)
    correct_orig = 0
    correct_adv = 0
    fooled = 0

    for i in range(0, total, batch_size):
        advs, ims, lbls = adv_examples[i:i+batch_size], originals[i:i+batch_size], gt_labels[i:i+batch_size]
        advs, ims, lbls = advs.cuda(), ims.cuda(), lbls.cuda()

        outputs_adv = net(advs)
        outputs_orig = net(ims)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        _, predicted_orig = torch.max(outputs_orig.data, 1)

        correct_adv += (predicted_adv == lbls).sum()
        correct_orig += (predicted_orig == lbls).sum()
        fooled += (predicted_adv != predicted_orig).sum()

    return (100.0 * float(fooled.item())/total, 100.0 * float(correct_adv.item())/total, 100.0 * float(correct_orig.item())/total)


# evaluates a set of adversarial examples against a set of models, reporting relevant metrics
# input: adv_examples: torch.tensor with size Nx3x32x32
#        gt_labels: torch.tensor with size N
#        originals: torch.tensor with size Nx3x32x32
#        network_list: list of Network class names and checkpoint paths on which adversarial examples will be tested
# returns:
# [(model1, fooling_ratio, accuracy_perturbed, accuracy_original),
#  (model2, fooling_ratio, accuracy_perturbed, accuracy_original),
#  ...]
def test_adv_examples_across_models(adv_examples,
                                    originals,
                                    gt_labels,
                                    network_list=[(ResNet18, '/share/cuvl/weights/cifar10/resnet18_epoch_347_acc_94.77.pth'),
                                                  (DenseNet121, '/share/cuvl/weights/cifar10/densenet121_epoch_315_acc_95.61.pth'),
                                                  (GoogLeNet, '/share/cuvl/weights/cifar10/googlenet_epoch_227_acc_94.86.pth'),
                                                  (SENet18, '/share/cuvl/weights/cifar10/senet18_epoch_279_acc_94.59.pth')],
                                    batch_size=4):
    accum = []

    for (network, weights_path) in network_list:
        net = network().cuda()
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        net.eval()

        res = test_adv_examples_for_model(adv_examples, originals, gt_labels, net)
        res = (weights_path, res[0], res[1], res[2])
        accum.append(res)

    return accum

# evaluates a set of adversarial examples against a set of models, reporting relevant metrics
# input: adv_examples: torch.tensor with size Nx3x32x32
#        gt_labels: torch.tensor with size N
#        originals: torch.tensor with size Nx3x32x32
#        network_list: list of Network class names and checkpoint paths on which adversarial examples will be tested
# returns:
# [(model1, fooling_ratio, accuracy_perturbed, accuracy_original),
#  (model2, fooling_ratio, accuracy_perturbed, accuracy_original),
#  ...]
def test_adv_examples_across_full_models(adv_examples,
                                    originals,
                                    gt_labels,
                                    network_list=[],
                                    batch_size=4):
    accum = []
    counter = 1
    for (network, name) in network_list:
        net = network.cuda()
        net.eval()

        res = test_adv_examples_for_model(adv_examples, originals, gt_labels, net)
        res = (name, res[0], res[1], res[2])
        accum.append(res)
        print('Adversarial example evaluated on transfer model '+str(counter)+'.')
        counter = counter + 1

    return accum