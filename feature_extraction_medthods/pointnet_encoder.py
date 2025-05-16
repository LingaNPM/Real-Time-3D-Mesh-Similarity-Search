import torch
import torch.nn as nn

class SimplePointNet(nn.Module):
    def __init__(self):
        super(SimplePointNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):  # x: [B, 3, N]
        x = self.mlp(x)
        return x.view(x.size(0), -1)  # [B, 1024]
