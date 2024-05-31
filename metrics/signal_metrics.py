import torch
import torch.nn as nn


class SignalMetrics(nn.Module):
    def __init__(self, config):
        super(SignalMetrics, self).__init__()
        self.config = config
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        l1_value = self.l1_loss(prediction.view(-1), target)
        return {
            "l1_loss": l1_value.item(),
        }

