import torch

from model.memoryMoudle import MemoryMoudle
from model.msffMoudle import MsffMoudle
from model.spatialAttention import SpatialAttentionMoudel
from model.utils.resnet18 import resNet18
import torch.nn as nn
import torch.nn.functional as F


class Memseg(nn.Module):
    def __init__(self, MI):
        super(Memseg, self).__init__()
        self.extractor0, self.extractor1, self.extractor2, self.extractor3, self.extractor4 = resNet18()
        self.memory = MemoryMoudle(MI)
        self.msff = MsffMoudle()
        self.attention = SpatialAttentionMoudel()
        self.upConv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.upConv2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upConv3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upConv4 = nn.Sequential(
            nn.Conv2d(128, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.upConv5 = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # 1应该是2的
            nn.Conv2d(48, 1, 3, padding=1)
        )
        self.conv1 = nn.Conv2d(64, 48, 3, padding=1)

    def forward(self, x):
        feature0 = self.extractor0(x)
        feature1 = self.extractor1(feature0)
        feature2 = self.extractor2(feature1)
        feature3 = self.extractor3(feature2)
        feature4 = self.extractor4(feature3)
        CI1, CI2, CI3, MI = self.memory(feature1, feature2, feature3)
        msff1, msff2, msff3 = self.msff(CI1, CI2, CI3)
        attention1, attention2, attention3 = self.attention(msff1, msff2, msff3, MI)
        _, _, H, W = attention3.size()
        upsample1 = F.upsample(feature4, size=(H, W), mode='bilinear')
        upConv1 = self.upConv1(upsample1)
        out1 = torch.cat([upConv1, attention3], dim=1)
        #
        _, _, H, W = attention2.size()
        upsample2 = F.upsample(out1, size=(H, W), mode='bilinear')
        upConv2 = self.upConv2(upsample2)
        out2 = torch.cat([upConv2, attention2], dim=1)
        #
        _, _, H, W = attention1.size()
        upsample3 = F.upsample(out2, size=(H, W), mode='bilinear')
        upConv3 = self.upConv3(upsample3)
        out3 = torch.cat([upConv3, attention1], dim=1)
        #
        conv1 = self.conv1(feature0)
        _, _, H, W = conv1.size()
        upsample4 = F.upsample(out3, size=(H, W), mode='bilinear')
        upConv4 = self.upConv4(upsample4)
        out4 = torch.cat([upConv4, conv1], dim=1)
        upsample5 = F.upsample(out4, size=(256, 256), mode='bilinear')
        out5 = self.upConv5(upsample5)
        return out5

