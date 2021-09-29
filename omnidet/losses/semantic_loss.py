"""
Semantic Segmentation loss functions for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
from torch import nn


class CrossEntropyLoss2d(nn.Module):
    """This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class."""

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, output, target):
        logpt = self.loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
