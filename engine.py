import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.conv_net import ConvNet
from data_loader.data_loader import CustomDataset

import pandas as pd
import sys

def train_one_epoch(model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                    data_loader: DataLoader, device: torch.device, epoch: int, epochs: int):
    
    print(f"Epoch: [{epoch:03d}/{epochs:03d}]")
    
    model.train()
    
    for batch_idx, (batches, targets) in enumerate(data_loader, start=1):
        batches = batches.to(device)
        targets = targets.to(device)
        
        # Feed-Forward
        optimizer.zero_grad()
        logits = model(batches)
        loss = loss_fn(logits, targets)
        
        # Back Propagation
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(data_loader):.2f}% Loss: {loss.item():.4f}", end="")
    print()


def evaluate(model: nn.Module, loss_fn: nn.Module, data_loader: DataLoader,
             device: torch.device):
    
    model.eval()
    
    total_num = 0
    correct_num = 0
    
    total_loss = 0.
    
    with torch.no_grad():
        for batch_idx, (batches, targets) in enumerate(data_loader, start=1):
            batches = batches.to(device)
            targets = targets.to(device)
            
            logits = model(batches)
            loss = loss_fn(logits, targets)
            
            total_loss += loss.item()
            
            outputs = F.softmax(logits, dim=-1)
            outputs = torch.argmax(outputs, dim=1)
            
            correct_num += (outputs == targets).sum()
            total_num += len(outputs)
            print(f"\rTest: {100*batch_idx/len(data_loader):.2f}%", end="")
    
    print()
    print(f"Accuracy: {100*correct_num/total_num:.2f}%, Test Loss: {total_loss/len(data_loader):.4f}")


if __name__ == '__main__':
    model = ConvNet()
    ds = CustomDataset(pd.read_csv('data\mnist.csv'))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_one_epoch(model, loss_fn, optimizer, dl, 'cpu', 1)
    evaluate(model, loss_fn, dl, 'cpu')