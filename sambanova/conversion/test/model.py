import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 5) 
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        loss = self.criterion(out,x)
        return loss, out

        
        