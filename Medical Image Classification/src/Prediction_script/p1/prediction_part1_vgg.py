import warnings
warnings.filterwarnings("ignore")
import sys
import os
import torch
import torchvision.models as models
import torch.nn as nn
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

def load_resnet_model(model_path, pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
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
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    with open("results.txt", "w") as f:
        for label, pred in zip(all_labels, all_predictions):
            if(pred == 0):
                f.write(f"negative\n")
            else:
                f.write(f"positive\n")
    return accuracy, precision, recall, f1

model_folder = "./models/1/vgg/"
model_name = "vgg.pth"
model = load_resnet_model(model_folder + model_name, pretrained=False)
accuracy, precision, recall, f1 = evaluate(model, test_loader)
print(f"{accuracy:.4f}   & {precision:.4f}   & {recall:.4f}   & {f1:.4f}")
