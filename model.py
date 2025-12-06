import torch
import torch.nn as nn
import torch.nn.functional as F

class HexaPawnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, 14)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy_out = self.policy_head(x) # policy logits
        value_out = torch.tanh(self.value_head(x))
        
        return policy_out, value_out