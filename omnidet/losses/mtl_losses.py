"""
Implements Uncertainty MTL loss function for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    def __init__(self, tasks=None):
        """Multi-task learning  using uncertainty to weigh losses for scene geometry and semantics.
        A specific type of uncertainty that does not change with input data and is task-specific, we learn this
        uncertainty and use it to down weigh each task. Note that, increasing the noise parameter σ reduces the
        weight for the task. Larger the uncertainty, smaller the contribution of the task’s loss to total loss.
        Consequently, the effect of task on the network weight update is smaller when the task’s homoscedastic
        uncertainty is high.
        Changed from log(sigma[idx].pow(2)) to 1 + self.sigma[idx].pow(2))
        In order to enforce positive regularization values. Thus, decreasing sigma to sigma.pow(2) < 1
        no longer yields negative loss v
        """
        super().__init__()
        self.tasks = tasks.split('_')
        self.sigma = nn.Parameter(torch.ones(len(self.tasks)), requires_grad=True)

    def forward(self, losses):
        loss = 0
        for idx, current_task in enumerate(self.tasks):
            loss += (1 / (2 * self.sigma[idx].pow(2))) * losses[f"{current_task}_loss"] + torch.log(
                1 + self.sigma[idx].pow(2))
            losses[f"sigma/{current_task}"] = self.sigma[idx]
            losses[f"sigma/{current_task}_weightage"] = 1 / (2 * self.sigma[idx].pow(2))
        return loss
