
import numpy as np
import torch
from torch import nn

from ..utils_model import sandwich

class TPPNet(nn.Module):
    def __init__(self, data_adapters, input_size, hidden_size, output_size):
        super(TPPNet, self).__init__()
        self.input_adapter = data_adapters["input_adapter"]
        self.output_adapter = data_adapters["output_adapter"]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, param_type: torch.Tensor, param_timeinterval: torch.Tensor, history_embedding: torch.Tensor, t: torch.Tensor):
        flat_x = self.input_adapter(param_type, param_timeinterval, history_embedding, t)
        x = torch.relu(self.fc1(flat_x)) 
        out = self.fc2(x)
        out = self.output_adapter(out)
        return out