import torch
import torch.nn as nn


class BPRMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        m1 = nn.Softmax(dim=1)
        m2 = nn.Sigmoid()
        ri = torch.gather(input, 1, target)
        sigmoids = m2(ri - input)
        expits = m1(input)
        scores_exp = torch.sum(expits * sigmoids, dim=1)
        return -torch.log(scores_exp)
