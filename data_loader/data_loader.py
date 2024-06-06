import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, df=pd.DataFrame, transform=None):
        super().__init__()
        
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        data = self.df.iloc[index].to_numpy()
        
        label = data[0]
        sample = data[1:].reshape(1, 28, 28).astype(np.float32)
        
        # Normalize
        sample /= 255.0
        
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64) # int64(long) for CrossEntropy
        
        return sample, label