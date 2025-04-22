import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PCamDataset(Dataset):
    def __init__(self, h5_file_x, h5_file_y, transform=None):
        self.h5_x = h5py.File(h5_file_x, 'r')['x']
        self.h5_y = h5py.File(h5_file_y, 'r')['y']
        self.transform = transform
    
    def __len__(self):
        return len(self.h5_x)
    
    def __getitem__(self, idx):
        img = self.h5_x[idx] / 255.0  # Normalize image
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) 

        label = torch.tensor(self.h5_y[idx], dtype=torch.long).item()  
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),         
    transforms.RandomVerticalFlip(),          
    transforms.RandomRotation(degrees=15),     
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
])