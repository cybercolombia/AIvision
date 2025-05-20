import torch
import torch.nn as nn

# Logistic clasifier for points in a 2D space and num_classes classes
class LogClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layer = nn.Linear(2, num_classes) #in: 2 features, out: num_classes classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = torch.sigmoid(self.layer(x))
        loss = self.criterion(out, y)
        return loss, out

        
        