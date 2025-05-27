import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

from model import AutoEncoder

DATA_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/aenet/MNISTData/'
MODEL_STORE_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/aenet/pytorch_models/'


def prepare_dataloader(batch_size_train: int, batch_size_test: int) -> Tuple[DataLoader, DataLoader]:    
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader


def train(epoch: int, log_interval: int, batch_size_train: int,
          model: nn.Module, train_loader: DataLoader,
          optimizer: torch.optim.Adam, 
          train_losses: List[float], train_counter: List[int],
          device: str) -> None:
    model.train() #tell the model it is a training round
    for batch_idx, data in enumerate(train_loader):
        x_batch, _ = data
        x_batch = x_batch.reshape(-1, 28*28).to(device)
        
        loss, pred = model(x_batch)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(x_batch), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))
            train_losses.append(loss.item())
            train_counter.append(batch_idx*batch_size_train + epoch*len(train_loader.dataset))

def test(model: nn.Module, test_loader: DataLoader,
        test_losses: List[float], device: str) -> None:
    model.eval() #tell the model it is being evaluated
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x_batch, _ = data
            x_batch = x_batch.reshape(-1, 28*28).to(device)
            loss, pred = model(x_batch)
            test_loss += loss.item()
            val = pred.data.max(1, keepdim=True)[1]
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}'.format(test_loss,))


def main():
    # Hyperparameters
    epochs = 6
    learn_rate = 1e-3
    batch_size_train = 32
    batch_size_test = 1000

    log_interval = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    
    train_loader, test_loader = prepare_dataloader(batch_size_train, batch_size_test)
    
    model = AutoEncoder().to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=learn_rate,
                                weight_decay=1e-8)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    
    # Train the model
    test(model, test_loader, test_losses, device)
    for epoch in range(epochs):
        train(epoch, log_interval, batch_size_train,
              model, train_loader, optimizer, 
              train_losses, train_counter, device)
        test(model, test_loader, test_losses, device)

    # Save the model
    torch.save(model, MODEL_STORE_PATH + 'aenet_model.ckpt')
    
    # Save training losses
    with open("train_losses.dat", "w") as f:
        for i in range(len(train_counter)):
            f.write(f"{train_counter[i]} {train_losses[i]}\n")

    # Save test losses
    with open("test_losses.dat", "w") as f:
        for i in range(len(test_counter)):
            f.write(f"{test_counter[i]} {test_losses[i]}\n")

if __name__ == '__main__':
    main()