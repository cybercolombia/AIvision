import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

DATA_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/aenet/MNISTData/'
MODEL_STORE_PATH = '/home/carlos/Documents/CyberComputingConsulting/AIvision/repo/AIvision/sambanova/conversion/aenet/pytorch_models/'

if __name__ == '__main__': 
    train_counter = []
    train_losses = []
    with open("train_losses.dat", "r") as f:
        for line in f:
            v = list(map(float, line.split()))
            train_counter.append(v[0])
            train_losses.append(v[1])
    
    test_counter = []
    test_losses = []
    with open("test_losses.dat", "r") as f:
        for line in f:
            v = list(map(float, line.split()))
            test_counter.append(v[0])
            test_losses.append(v[1])
    
    plt.plot(train_counter,train_losses,zorder=5)
    plt.scatter(test_counter,test_losses,color='r',zorder=15)
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    #plt.show()

    ##-------------------------------------------------------------

    model = torch.load(MODEL_STORE_PATH+'aenet_model.ckpt', weights_only=False).to('cpu')
    model.eval()
    
    batch_size_test = 1000
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
    
    with torch.no_grad():
        img_batch = next(iter(test_loader))
        img = img_batch[0][0][0]
        _, reconst = model(img.reshape(-1))
        reconst = reconst.reshape((28,28))
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(img, cmap='grey')
        ax = fig.add_subplot(1,2,2)
        ax.imshow(reconst, cmap='grey')
        plt.show()
    


