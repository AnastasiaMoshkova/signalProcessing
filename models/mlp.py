import torch.nn as nn
import torch.nn.functional as F

class MlpModel(nn.Module):
    def __init__(self, config):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(config['input_ch'],config['linear1'])
        self.linear2 = nn.Linear(config['linear1'], config['output_dim'])


    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out