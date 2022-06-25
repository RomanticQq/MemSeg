import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms,datasets
import os
import torch
from PIL import Image
import numpy as np
import cv2

dir_path = './dataset/traindata/default_gts'
name_list = os.listdir(dir_path)[0]
img_path = os.path.join(dir_path, name_list)
img = Image.open(img_path)
img = img.convert("")
img = np.asarray(img, dtype=np.float32)

print(img.shape)
