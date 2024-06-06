import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from models.conv_net import ConvNet
from data_loader.data_loader import CustomDataset

from engine import train_one_epoch, evaluate

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # Train or Inference
    parser.add_argument("--mode", type=str, default='train')
    
    # Hyperparameter
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    
    # utils
    parser.add_argument("--device", type=str, default='cpu')
    
    # Saved Model Location
    parser.add_argument("--file_name", type=str)
    
    # Inference Data Index
    parser.add_argument("--index", type=int, default=0)
    return parser


def main(args):
    # Device Utilization
    device = None
    if args.device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    if args.mode == 'train':
        # Load Model
        model = ConvNet().to(device)
        print('Load Model Complete')
        # Load Loss Function
        loss_fn = nn.CrossEntropyLoss()
        # Load Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Load Dataset
        ds = CustomDataset(df=pd.read_csv('data\mnist.csv'))
        print('Load Dataset Complete')
        # Train, Test Split
        train_size = int(len(ds) * 0.7)
        test_size = len(ds) - train_size
        train_ds, test_ds = random_split(ds, [train_size, test_size])
        # Load DataLoader
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
        
        # Train
        print('====================')
        for epoch in range(args.epochs):
            train_one_epoch(model, loss_fn, optimizer, train_dl, device, epoch+1, args.epochs)
            evaluate(model, loss_fn, test_dl, device)
            print('--------------------')
            
        # Save Model
        if args.file_name:
            try:
                torch.save(model.state_dict(), os.path.join('saved', args.file_name))
                print("Saving Model Success!")
            except:
                print("Saving Model Failed...")
                
    elif args.mode == 'inference':
        # Load Trained Model
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(os.path.join('saved', args.file_name)))
        model.eval()
        print('Load Model Complete')
        
        # Load Dataset
        ds = CustomDataset(df=pd.read_csv('data\mnist.csv'))
        print('Load Dataset Complete')
        
        sample, target = ds[args.index]
        logit = model(sample.reshape(1, *sample.shape))
        
        output = F.softmax(logit, dim=-1)
        output = torch.argmax(output).item()
        
        img = sample.cpu().detach().numpy().reshape(28, 28)
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Predict {output}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ConvNet-MNIST training and evaluation Script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)