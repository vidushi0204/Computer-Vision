import os
import sys
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
warnings.filterwarnings("ignore")


def main(image_dir, checkpoint_path, output_predictions):
    # Load base pretrained model + processor
    base_model_name = "SenseTime/deformable-detr"
    model = DeformableDetrForObjectDetection.from_pretrained(base_model_name)
    processor = DeformableDetrImageProcessor.from_pretrained(base_model_name)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    val_ann_path = os.path.join(os.path.dirname(image_dir), 'annotations_val.json')
    with open(val_ann_path, 'r') as f:
        coco = json.load(f)
    id2filename = {img['id']: img['file_name'] for img in coco['images']}

    os.makedirs(os.path.dirname(output_predictions), exist_ok=True)
    coco_results = []

    for img_id in tqdm(id2filename):
        img_path = os.path.join(image_dir, id2filename[img_id])
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=target_sizes)[0]
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x0, y0, x1, y1 = box.tolist()
            coco_box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": coco_box,
                "score": float(score)
            })

    with open(output_predictions, "w") as f:
        json.dump(coco_results, f)
    print(f"Saved detections to {output_predictions}")

    def evaluate_with_cocoeval(gt_ann_path, dt_path):
        coco_gt = COCO(gt_ann_path)
        coco_dt = coco_gt.loadRes(dt_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    evaluate_with_cocoeval(val_ann_path, output_predictions)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 1_2.py <image_dir> <checkpoint_path.pth> <output_predictions>")
        sys.exit(1)

    image_dir = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_predictions = sys.argv[3]
    main(image_dir, checkpoint_path, output_predictions)
