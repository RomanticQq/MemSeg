import torch
import torch.nn as nn
import torch.nn.functional as F


class CA_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CA_Block, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1x1 = nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.F_h = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = s_h.expand_as(x) * s_w.expand_as(x)
        return out


class MsffMoudle(nn.Module):
    def __init__(self):
        super(MsffMoudle, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(256,256, 3, padding=1)
        self.conv3 = nn.Conv2d(512,512, 3, padding=1)
        self.ca_block1 = CA_Block(128)
        self.ca_block2 = CA_Block(256)
        self.ca_block3 = CA_Block(512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1)
        )
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 64, 3, padding=1)

    def forward(self, CI1, CI2, CI3):
        attention1 = self.conv1(CI1) * self.ca_block1(CI1)
        attention2 = self.conv2(CI2) * self.ca_block2(CI2)
        attention3 = self.conv3(CI3) * self.ca_block3(CI3)
        out1 = self.conv4(attention1)
        out2 = self.conv5(attention2)
        msff3 = self.conv6(attention3)
        _, _, H, W = out2.size()
        msff2 = self.conv7(F.upsample(msff3, size=(H, W), mode='bilinear')) + out2
        _, _, H, W = out1.size()
        msff1 = self.conv8(F.upsample(msff2, size=(H, W), mode='bilinear')) + out1
        return msff1, msff2, msff3


