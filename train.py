import torch
from model.memseg import Memseg
from model.utils import resnet18
from torch.utils import data as data_
from data.dataset import Dataset
from config import opt
from model.utils.focus_loss import focus_loss


def train():
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False,num_workers=1)
    MI = resnet18.create_mi()
    model = Memseg(MI)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    for epoch in range(100):
        loss_total = 0.0
        for ii, (img, label) in enumerate(dataloader):
            model.train()
            out = model(img)
            # print(out.shape)
            # print(label.shape)
            # exit()
            loss = torch.dist(out, label, p=1) + focus_loss(15, out, label)
            optimizer.zero_grad()
            loss.backward()
            loss_total = loss_total + loss
        loss = loss_total / len(dataset)
        print("当前epoch：{}, loss: {.4f}".format(epoch, loss))




if __name__ == '__main__':
    train()
