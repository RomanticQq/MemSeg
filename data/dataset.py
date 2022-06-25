import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def read_image(path, color=True):
    f = Image.open(path)
    img = f.convert("RGB")
    if hasattr(f, 'close'):
        f.close()
    img = np.asarray(img, dtype=np.float32)
    img = img.transpose((2, 0, 1))
    return img


def proprocess(img):
    img = img / 255.
    normalize = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize([256, 256])
    ])
    return normalize(torch.from_numpy(img)).numpy()


def gtPreprocess(img):
    resize = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Grayscale(num_output_channels=1)
    ])
    img = resize(torch.from_numpy(img)).numpy()
    return img


def transfrom(img):
    img = proprocess(img)
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1)
    ])
    img = transform(torch.from_numpy(img)).numpy()
    return img


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.train_name_list = os.listdir(self.opt.train_dir)
        self.val_name_list = os.listdir(self.opt.val_dir)

    def __getitem__(self, idx):
        train_path = os.path.join(self.opt.train_dir, self.train_name_list[idx])
        val_path = os.path.join(self.opt.val_dir, self.val_name_list[idx])
        train_img = read_image(train_path)
        val_img = read_image(val_path)
        train_img = transfrom(train_img)
        val_img = gtPreprocess(val_img)
        return train_img.copy(), val_img.copy()

    def __len__(self):
        return len(self.train_name_list)