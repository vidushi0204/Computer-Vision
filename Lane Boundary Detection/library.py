import numpy as np
import cv2
from funcs import *
from cv2 import HoughLinesP
from poly import draw
def red_image(img):
    hls = bgr_to_hls(img)

    brown_mask = inrange(hls, [50, 50, 100], [25, 150, 200])

    gray = bgr_to_gray(img)
    gray = gaussian_blur(gray, 7, 1)
    gray = np.clip(gray - brown_mask, 0, 255)
    sobelx = sobel(gray, 1, 0, ksize=5) 
 
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_mask = inrange(scaled_sobel, [40], [150])
    grad_mask = np.clip(grad_mask - brown_mask, 0, 255)
    preprocessed = gaussian_blur(grad_mask, 9, 2)

    return preprocessed

def grass_image(img, white_threshold=230):
    gray = bgr_to_gray(img)

    white_mask = (
        (img[:,:,2] > white_threshold) &  
        (img[:,:,1] > white_threshold) & 
        (img[:,:,0] > white_threshold) 
    )
    gray[~white_mask] = 0

    preprocessed = gaussian_blur(gray, 9, 2)
    return preprocessed


def preprocess_image(img):

    hls = bgr_to_hls(img)
    brown_mask = inrange(hls, [10,30,40], [50,150,255])
    brown_percentage = (count_nonzero(brown_mask) / brown_mask.size) * 100

    if(brown_percentage<1):
        return grass_image(img)
    else:
        return red_image(img)
    
def detect_edges(preprocessed_img):
    fin = canny(preprocessed_img, 100, 200)
    return fin



def filter_nearby_lines(lines, distance_thresh = 80, angle_thresh = 20):
    if lines is None or len(lines) == 0:
        return []

    lines = [line[0] for line in lines]  

    def cartesian_to_polar(line):
        """Convert Cartesian (x1, y1, x2, y2) to polar (rho, theta)."""
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        theta = np.arctan2(dy, dx) * (180 / np.pi) 
        rho = abs(x1 * dy - y1 * dx) / np.hypot(dx, dy)
        return rho, theta, line

    polar_lines = sorted(
        [cartesian_to_polar(line) for line in lines],
        key=lambda x: np.linalg.norm([x[2][2] - x[2][0], x[2][3] - x[2][1]]),
        reverse=True
    )

    filtered_lines = []
    used = set()

    for i, (rho1, theta1, line1) in enumerate(polar_lines):
        if i in used:
            continue 
        
        for j, (rho2, theta2, line2) in enumerate(polar_lines[i+1:], start=i+1):
            if j in used:
                continue 

            angle_diff = abs(theta1 - theta2)
            angle_diff = min(angle_diff, 180 - angle_diff)
            if abs(rho1 - rho2) < distance_thresh and angle_diff < angle_thresh:
                used.add(j) 
            elif line_intersection(line1, line2):
                used.add(j)
    for i, (rho1, theta1, line) in enumerate(polar_lines):
        if i in used:
            continue 
        filtered_lines.append(line)
    return np.array(filtered_lines, dtype=np.int32).reshape(-1, 1, 4)

def filter_nonroi_lines(lines, image_height):
    if lines is None or len(lines) == 0:
        return []

    filtered_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line points

        if top((x1, y1, x2, y2), image_height) or (bottom((x1, y1, x2, y2), image_height) and angle_with_horizontal((x1, y1, x2, y2)) < 30):
            continue 

        filtered_lines.append(line)

    return np.array(filtered_lines, dtype=np.int32) if filtered_lines else None


        
def hough_lines_p(edges: np.ndarray, rho: float = 1, theta: float = np.pi / 180, threshold: int = 80, minLineLength: int = 290):
    diagonal = np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)

    theta_angles = np.arange(0, np.pi, theta)  
    rho_values = np.arange(-diagonal, diagonal, rho)
    
    accumulator = np.zeros((len(rho_values), len(theta_angles)), dtype=int)
    
    sins = np.sin(theta_angles)
    coss = np.cos(theta_angles)
    xs, ys = np.where(edges > 0) 
    for x, y in zip(xs, ys):
        for t in range(len(theta_angles)):
            current_rho = x * coss[t] + y * sins[t]
            rho_pos = np.argmin(np.abs(current_rho - rho_values))
            accumulator[rho_pos, t] += 1
    
    lines = []
    for r_idx, t_idx in zip(*np.where(accumulator > threshold)):
        rho_val = rho_values[r_idx]
        theta_val = theta_angles[t_idx]
        
        x0 = int(rho_val * np.cos(theta_val))
        y0 = int(rho_val * np.sin(theta_val))
        
        x1 = int(x0 + 1000 * (-np.sin(theta_val)))
        y1 = int(y0 + 1000 * (np.cos(theta_val)))
        x2 = int(x0 - 1000 * (-np.sin(theta_val)))
        y2 = int(y0 - 1000 * (np.cos(theta_val)))
        
        if minline(x1,x2,y1,y2,minLineLength):
            lines.append([[x1, y1, x2, y2]])
    return np.array(lines)

def lane_detection(edges, img):
    lines = HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, maxLineGap=40, minLineLength=290)
    lines = filter_nonroi_lines(lines, img.shape[0])
    lines = filter_nearby_lines(lines)

    if(len(lines)==0):
            preprocessed_image = grass_image(img,200)
            edges = detect_edges(preprocessed_image)
            lines = HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=70, maxLineGap=50, minLineLength=120)
            lines = filter_nonroi_lines(lines, img.shape[0])
            lines = filter_nearby_lines(lines)
    return lines

def detect_lane_boundaries(edges, img):
    lines = lane_detection(edges, img)
    hls = bgr_to_hls(img)
    brown_mask = inrange(hls, [10,30,40], [50,150,255])
    brown_percentage = (count_nonzero(brown_mask) / brown_mask.size) * 100
    if(brown_percentage<1):
        lines = lane_detection(edges, img)
            
    else:
        lines = HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=70, maxLineGap=50, minLineLength=120)
        lines = filter_nonroi_lines(lines, img.shape[0])
        lines = filter_nearby_lines(lines, distance_thresh=20, angle_thresh=15)
    
    output_image = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)


    return output_image

def find_intersection(line1, line2):
    a1, b1, a2, b2 = line1[0]
    x1, y1, x2, y2 = line2[0]
    det = (a1-a2)*(y1-y2)-(b1-b2)*(x1-x2)
    
    if det==0: 
        return None
    
    m = ((a1-x1)*(y1-y2)-(b1-y1)*(x1-x2))/det
    # c = ((b1-b2)*(a1-x1)-(a1-a2)*(b1-y1))/det

    return [a1+m*(a2-a1), b1+m*(b2-b1)]

def compute_line_fit(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            op = find_intersection(lines[i],lines[j])
            if op is not None:
                intersections.append(op)
    # print(intersections)
    if len(intersections)==0:
        return 0.0

    C_x = np.mean([x for x, y in intersections])
    C_y = np.mean([y for x, y in intersections])

    dist = 0.0
    for x, y in intersections:  
        dist += np.sqrt((x - C_x)**2 + (y - C_y)**2)

    return dist

