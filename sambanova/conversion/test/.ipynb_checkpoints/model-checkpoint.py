import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3) 
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = torch.sigmoid(self.layer(x))
        loss = self.criterion(out, y)
        return loss, out

        
        