import PIL
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import brier_score_loss
from torch.autograd import Variable
import math
import argparse

def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)

def get_entropy(out_prob):

    log_probs = torch.log(out_prob)
    U = (out_prob * log_probs).sum(1)
    max, _ = torch.max(out_prob, 1)
    return U, max

def get_purview(img, outputs, model):

    #loss_label = [[1 1... 1]]
    loss_label = torch.from_numpy(1/1 * np.ones(1000)).float().unsqueeze(0)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    gradients = []
    model = model.eval().cuda()
    img = img.cuda()
    loss_arg = torch.tensor(loss_label * img.shape[0]).float().cuda()
    outputs = model(img)
    loss = loss_fn(outputs, loss_arg)
    loss.mean().backward(retain_graph=False)

    for j, (n, p) in enumerate(model.named_parameters()):
        gradients.append((p.grad**2).data.sum().cpu().numpy())

    gradients_initial = gradients[:math.floor(j/2)]
    gradients_final = gradients[math.ceil(j/2):]

    gradients_initial = -np.mean(gradients_initial)
    gradients_final = -np.mean(gradients_final)

    return gradients_initial, gradients_final

def get_gradnorm(img, outputs, model):

    zero_grad(model)

    model = model.eval().cuda()
    outputs = model(img.cuda())

    loss_label = torch.from_numpy(1 / 1000 * np.ones(1000)).float().unsqueeze(0)
    kl_loss = nn.KLDivLoss()

    out_prob = F.softmax(outputs, dim=1).cuda()

    loss = kl_loss(loss_label.cuda(), out_prob)
    loss.backward(retain_graph=False)

    for j, (n, p) in enumerate(model.named_parameters()):
        if len(p.shape) == 2:
            if p.size(0) == 1000:
                grads = p.grad
                break
        else:
            continue

    return -torch.norm(grads, p = 2)

def get_margin(out_prob):

    probs_sorted, idxs = out_prob.sort(descending=True)
    #probs_sorted, idxs = logits.sort(descending=True)
    U = probs_sorted[:, 0] - probs_sorted[:, 1]

    return U

def get_nll_brier(logits, out_prob, target):

    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    prob = np.asarray(out_prob.data.squeeze().cpu().numpy())
    y_true_temp = one_hot(np.asarray(target), 1000)
    brier_score = brier_score_loss(y_true_temp, prob)

    target = target.squeeze()
    pred = torch.from_numpy(np.asarray([target])).cuda()
    #logits = logits.squeeze(0)
    log_likelihood = -F.nll_loss(logits, pred)

    return log_likelihood, brier_score

def zero_grad(self):
    # ""Sets gradients of all model parameters to zero."""
    for p in self.parameters():
        if p.grad is not None:
            p.grad.data.zero_()

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_grads_individual(img, model, feat):

    torch.cuda.empty_cache()
    ce_loss2 = nn.MSELoss()
    zero_grad(model)
    model.cuda()
    model.eval()
    img = img.cuda()

    im_label_as_var2 = torch.from_numpy(1 / 1000 * np.ones(1000)).float()
    im_label_as_var2 = torch.unsqueeze(im_label_as_var2, 0)

    output = model(img)
    _, label = torch.max(output, 0)

    pred_loss = ce_loss2(output.cuda(), im_label_as_var2.cuda())
    pred_loss.backward(retain_graph=False)

    for j, (n, p) in enumerate(model.named_parameters()):
        if len(p.shape) == 2:
            if p.size(0) == 1000:
                temp_grad = p.grad
                break
        else:
            continue


    temp_grad = torch.unsqueeze(temp_grad, 0)
    temp_grad = temp_grad.data.cpu().squeeze()

    grad_energy_weights = (torch.var(torch.pow(temp_grad, 2), 1))
    grad_energy_max, label = torch.max(grad_energy_weights, 0)
    grad_energy_weights_mean = torch.mean(grad_energy_weights, 0)

    grad_trust = (grad_energy_max / grad_energy_weights_mean).data.cpu().numpy()
    zero_grad(model)

    del temp_grad, pred_loss, im_label_as_var2, output
    torch.cuda.empty_cache()

    return grad_trust

def get_ODIN(input, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    zero_grad(model)

    inputs = Variable(input, requires_grad=True).cuda()
    outputs = model(inputs.cuda())

    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def get_MCD(img, model, forward_passes, n_classes):

    zero_grad(model)

    def enable_dropout(model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    dropout_predictions = np.empty((0, 1, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        img = img.to(torch.device('cuda'))
        with torch.no_grad():
            output = model(img)
            output = softmax(output)  # shape (n_samples, n_classes)
        predictions = np.vstack((predictions, output.data.cpu().numpy()))

    dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))

    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    #variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = 0.0001
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)

    return entropy

def get_comparison_stats(img, model):

    torch.cuda.empty_cache()
    model.eval()

    img = img.cuda()
    model = model.cuda()

    output = model(img)
    _, prediction = torch.max(output, 1)
    prob = nn.Softmax()
    out_prob = prob(output).data.cpu()

    prediction = prediction.data.cpu().numpy()

    entropy, confidence = get_entropy(out_prob)
    max_margin = get_margin(out_prob)
    NLL, Brier = get_nll_brier(output, out_prob, prediction)
    ODIN_score = np.max(get_ODIN(img, output, model, 100, 0.0014)) #0.0014
    MCD_score = -get_MCD(img, model, 10, 1000)
    grad_norm_temp = get_gradnorm(img, output, model)
    purview_initial_temp, purview_final_temp = get_purview(img, output, model)
    proposed = get_grads_individual(img, model, 1000)


    return prediction, confidence.data.cpu().numpy(), entropy.data.cpu().numpy(), max_margin.data.cpu().numpy(), NLL.data.cpu().numpy(), Brier, ODIN_score, MCD_score, purview_initial_temp, purview_final_temp, grad_norm_temp.data.cpu().numpy(), proposed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='Neural network used at inference. Imported from torchvision v0.14 as torchvision.models.get_models(x, weights="DEFAULT"). Type in x...'
                                          'Networks that have been tested: alexnet, convnext, resnet18, resnet50, efficientnet_v2_s, maxvit_t,...'
                                          'mobilenet_v3_small, wide_resnet50_2, resnext50_32x4d, resnext101_64x4d, swin_b, swin_v2_c, ...'
                                          'vgg11_bn, vit_b_16', type=str, default='vit_b_16')
    opts = parser.parse_args()

    img_name = 'images/water-bird.JPEG'
    pil_img = PIL.Image.open(img_name)

    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalize(torch_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normed_torch_img.cuda()

    # Code checked on 14 networks pretrained and downloaded from "https://pytorch.org/vision/stable/models.html". Type in the names exactly as defined in PyTorch torchvision library
    # Only works on TorchVision v0.14. Otherwise use torchvision.models.
    model = models.get_model(opts.network, weights="DEFAULT").cuda()

    prediction, confidence, entropy, margin, NLL, Brier, ODIN, MCD, purview_initial, purview_final, grad_norm, grad_trust = get_comparison_stats(normed_torch_img, model)
    print('The prediction is : ' + str(prediction) + ' with GradTrust: ' + str(grad_trust))
    print('Comparison Metrics:\nSoftmax Confidence: ' + str(confidence) + '\nEntropy: ' + str(entropy) + '\nMargin: ' + str(margin) +'\nLog-likelihood: ' + str(NLL) +'\nODIN: ' + str(ODIN) + '\nMC-Dropout: ' + str(MCD) + '\nPurview (Initial layers): ' + str(purview_initial) + '\nPurview (Final layers): ' + str(purview_final) + '\nGrad Norm: ' + str(grad_norm))


if __name__ == '__main__':

    main()