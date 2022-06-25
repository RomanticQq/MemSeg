import torch
import numpy as np


def computer_weights(piexlist):
    sum_list = sum(piexlist) # 先求和
    piexlist_w = [sum_list/x for x in piexlist]
    sum_w = sum(piexlist_w) # 再求和
    w_final = [x/sum_w for x in piexlist_w]
    return w_final


def focus_loss(num_classes, input_data, target, cuda=True):
    n, c, h, w = target.shape
    input_data = torch.softmax(input_data, dim=1)
    classes_mask = torch.zeros_like(input_data)
    classes_mask.scatter_(1, target, 1)
    input_data = torch.sum(input_data * classes_mask, dim=1)
    gamma = 4
    num_class_list = []
    for i in range(num_classes):
        num_class_list.append(torch.sum(target == i).item())
    weights_alpha = computer_weights(num_class_list)
    weights_alpha = torch.tensor(weights_alpha)
    weights_alpha = weights_alpha[target.view(-1)].reshape(n, c, h, w)
    if cuda:
        weights_alpha = weights_alpha.cuda()
    loss = -(weights_alpha * torch.pow((1-input_data), gamma) * torch.log(input_data))
    loss = torch.mean(loss)
    return loss


a = torch.randn(1,3,256,256)
b = torch.randn(1,3,256,256)
c = focus_loss(5, a, b)
print(c)
