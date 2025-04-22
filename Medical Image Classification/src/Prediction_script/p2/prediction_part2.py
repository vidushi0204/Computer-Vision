import warnings
warnings.filterwarnings("ignore")
import sys
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
test_pref = "./test_imgs/"

class PCamDatasetPNG(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(label_file, "r") as f:
            self.data = [line.strip().split() for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(os.path.join(self.image_folder, image_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

test_dataset = PCamDatasetPNG(test_pref, test_pref + "labels.txt", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return self.conv(x) + self.skip(x)

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.conv3x3 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x)], dim=1)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 128, 2)
        self.res3 = ResidualBlock(128, 256, 2)

        self.inception = InceptionModule(256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
                
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.inception(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_custom_model(model_path):
    model = CustomCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    with open("results_2.txt", "w") as f:
        for label, pred in zip(all_labels, all_predictions):
            if(pred == 0):
                f.write(f"negative\n")
            else:
                f.write(f"positive\n")
    return accuracy, precision, recall, f1

model_folder = "./models/2/"
model_name = "cnn.pth"
model = load_custom_model(model_folder + model_name)
accuracy, precision, recall, f1 = evaluate(model, test_loader)
print(f"{accuracy:.4f}   & {precision:.4f}   & {recall:.4f}   & {f1:.4f}")
