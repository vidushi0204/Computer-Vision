import os
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
import sys
warnings.filterwarnings("ignore")

# Setup device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_ROOT = None
TRAIN_MODEL_PATH = None

processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(DEVICE)

# Control which layers to train
def set_trainable_layers(exp: int):
    for param in model.parameters():
        param.requires_grad = False
    if exp == 1:
        for param in model.parameters():
            param.requires_grad = True
    elif exp == 2:
        for name, param in model.named_parameters():
            if "decoder" in name:
                param.requires_grad = True
    elif exp == 3:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = True

# Dataset
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_path, processor):
        with open(ann_path, 'r') as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.processor = processor
        self.images = {img['id']: img for img in data['images']}
        self.ann_by_image = {img_id: [] for img_id in self.images}
        for ann in data['annotations']:
            self.ann_by_image[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        file_name = self.images[img_id]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        annotations = [{
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "area": ann["area"],
            "iscrowd": ann.get("iscrowd", 0)
        } for ann in self.ann_by_image[img_id]]

        encoding = processor(images=image, annotations={"image_id": img_id, "annotations": annotations}, return_tensors="pt")
        encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}
        encoding['img_id'] = img_id
        encoding['file_name'] = file_name
        encoding['image'] = image
        return encoding

# Collate function
def collate_fn(batch):
    keys = batch[0].keys()
    return {k: [d[k] for d in batch] for k in keys}

# Training function
def train_model(exp_id=1, epochs=10, batch_size=2):
    print(f"Training Experiment {exp_id}...")
    set_trainable_layers(exp_id)
    model.train()
    train_dataset = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_PATH, processor)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixel_values = torch.stack(batch["pixel_values"]).to(DEVICE)
    
            def move_label_to_device(label_dict, device):
                return {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in label_dict.items()
                }
            
            flat_labels = [move_label_to_device(label, DEVICE) for labels in batch["labels"] for label in labels]

            outputs = model(pixel_values=pixel_values, labels=flat_labels)
            loss = outputs.loss
     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)

if __name__ == "__main__":
    DATASET_ROOT = sys.argv[1]
    TRAINED_MODEL_PATH = sys.argv[2]
    TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "./foggy_dataset_A3_train")
    TRAIN_ANN_PATH = os.path.join(DATASET_ROOT, "./annotations_train.json")

