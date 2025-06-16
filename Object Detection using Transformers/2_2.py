import os
import sys
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from groundingdino.util.inference import Model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"

def wrap_prompt(class_names):
    return [f"a {name}" for name in class_names]
def main(image_dir, checkpoint_path, output_predictions):
    # Read annotations
    val_ann_path = os.path.join(os.path.dirname(image_dir), 'annotations_val.json')
    with open(val_ann_path, 'r') as f:
        coco = json.load(f)

    id2filename = {img['id']: img['file_name'] for img in coco['images']}
    id2widthheight = {img['id']: (img['width'], img['height']) for img in coco['images']}
    id2anns = {}
    for ann in coco['annotations']:
        id2anns.setdefault(ann['image_id'], []).append(ann)

    class_names = [cat['name'] for cat in coco['categories']]
    class_name_to_id = {cat['name']: cat['id'] for cat in coco['categories']}

    # Load model with specified checkpoint
    model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=checkpoint_path, device='cpu')

    # Prepare output directory
    os.makedirs(os.path.dirname(output_predictions), exist_ok=True)
    coco_results = []

    for img_id in tqdm(id2filename):
        img_path = os.path.join(image_dir, id2filename[img_id])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        detections = model.predict_with_classes(
            image=image,
            classes=wrap_prompt(class_names),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        for box, score, label in zip(detections.xyxy, detections.confidence, detections.class_id):
            x0, y0, x1, y1 = box.tolist()
            coco_box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(label + 1),
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
    image_dir = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_predictions = sys.argv[3]
    main(image_dir, checkpoint_path, output_predictions)
