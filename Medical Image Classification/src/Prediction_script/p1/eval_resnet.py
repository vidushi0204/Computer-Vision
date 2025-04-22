import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/kaggle/input/read-data")
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from read_data import PCamDataset 
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
test_pref = "/kaggle/input/test-data/"  

test_dataset = PCamDataset(test_pref + "camelyonpatch_level_2_split_test_x.h5", 
                            test_pref + "camelyonpatch_level_2_split_test_y.h5")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def load_resnet_model(model_path, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
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
    np.savetxt("result.txt", all_predictions, fmt="%d")
    # Compute overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return accuracy, precision, recall, f1

model_folder = "/kaggle/input/model/"
model_name = "resnet.pth"

model = load_resnet_model(model_folder + model_name, pretrained=False)

accuracy, precision, recall, f1 = evaluate(model, test_loader)

print(f"{accuracy:.4f}   & {precision:.4f}   & {recall:.4f}   & {f1:.4f}")

