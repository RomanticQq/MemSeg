import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms,datasets
import os
import torch
from PIL import Image
import numpy as np
import cv2


def resNet18():
    model = resnet18(pretrained=True)
    for layer in model.layer1:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in model.layer2:
        for p in layer.parameters():
            p.requires_grad = False
    for layer in model.layer3:
        for p in layer.parameters():
            p.requires_grad = False

    layer0 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu
    )
    layer1 = nn.Sequential(
        model.maxpool,
        model.layer1
    )
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return layer0, layer1, layer2, layer3, layer4


def create_mi():
    MI = list()
    data_dir = './dataset/traindata/normal_data'
    data_transforms = transforms.Compose([
        transforms.Resize([256, 256])
    ])
    dir_list = os.listdir(data_dir)
    for i in range(10):
        mi = list()
        extractor0, extractor1, extractor2, extractor3, _= resNet18()
        path = os.path.join(data_dir,dir_list[i])
        img = cv2.imread(path)
        img = img[np.newaxis]
        img = torch.from_numpy(img).permute(0,3,1,2)
        img = data_transforms(img).float()

        feature0 = extractor0(img)
        feature1 = extractor1(feature0)
        feature2 = extractor2(feature1)
        feature3 = extractor3(feature2)
        # print(feature1.shape)
        # print(feature2.shape)
        # print(feature3.shape)
        # exit()
        mi.append(feature1)
        mi.append(feature2)
        mi.append(feature3)
        MI.append(mi)
    return MI

create_mi()
