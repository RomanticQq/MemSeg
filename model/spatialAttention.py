import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttentionMoudel(nn.Module):
    def __init__(self):
        super(SpatialAttentionMoudel, self).__init__()
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.SpatialAttention3 = SpatialAttention()

    def forward(self, msff1, msff2, msff3, MI):
        M3 = self.SpatialAttention3(MI[2])
        _, _, H, W = MI[1].size()
        M2 = F.upsample(M3, size=(H, W), mode='bilinear') * self.SpatialAttention2(MI[1])
        _, _, H, W = MI[0].size()
        M1 = F.upsample(M2, size=(H, W), mode='bilinear') * self.SpatialAttention1(MI[0])
        out1 = msff1 * M1
        out2 = msff2 * M2
        out3 = msff3 * M3
        return out1, out2, out3
