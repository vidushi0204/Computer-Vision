import cv2
import numpy as np
import os
from library import *

    
def task1(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            preprocessed_image = preprocess_image(image)

            edges = detect_edges(preprocessed_image)

            output_image = detect_lane_boundaries(edges, image)

           
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, output_image)
            print(f"Processed and saved: {output_path}")
            cv2.destroyAllWindows()
