import numpy as np
import cv2
import os
import csv
from library import *

def task2(input_dir, output_csv):
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            preprocessed = grass_image(img)
            edges = detect_edges(preprocessed)
            lines = lane_detection(edges,img)
            print(filename)
            if lines is not None and len(lines)>0:
                quality = compute_line_fit(lines)
                results.append([filename, quality])
            else:
                results.append([filename, None])

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Line Fit Quality'])
        writer.writerows(results)

