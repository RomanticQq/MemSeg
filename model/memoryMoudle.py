import torch
import torch.nn as nn


class MemoryMoudle(nn.Module):
    def __init__(self, MI):
        super(MemoryMoudle, self).__init__()
        self.MI = MI

    def forward(self, feature1, feature2, feature3):

        different = list()
        MI = self.MI
        for i in range(10):
            f1_diff = torch.dist(feature1, MI[i][0])
            f2_diff = torch.dist(feature2, MI[i][1])
            f3_diff = torch.dist(feature3, MI[i][2])
            total = f1_diff + f2_diff + f3_diff
            different.append(total)
        different = torch.stack(different)
        mi_index = torch.argmin(different)
        MI = MI[mi_index]
        CI1 = torch.cat([feature1, MI[0]], dim=1)
        CI2 = torch.cat([feature2, MI[1]], dim=1)
        CI3 = torch.cat([feature3, MI[2]], dim=1)
        return CI1, CI2, CI3, MI




