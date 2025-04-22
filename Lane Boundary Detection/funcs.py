import cv2
import numpy as np
def bgr_to_hls(img):
    img = img.astype(np.float32) / 255.0 
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    
    max_val = np.maximum(R, np.maximum(G, B))
    min_val = np.minimum(R, np.minimum(G, B))
    
    L = (max_val + min_val) / 2
    
    delta = max_val - min_val
    S = np.zeros_like(L)
    
    mask = delta > 0
    S[mask] = delta[mask] / (1 - np.abs(2 * L[mask] - 1))
    
    H = np.zeros_like(L)
    
    mask_r = (max_val == R) & mask
    mask_g = (max_val == G) & mask
    mask_b = (max_val == B) & mask
    
    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / delta[mask_r]) + 360) % 360
    H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120) % 360
    H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240) % 360
    
    H = (H / 2).astype(np.uint8) 
    L = (L * 255).astype(np.uint8)
    S = (S * 255).astype(np.uint8)
    
    hls_image = np.stack([H, L, S], axis=-1)
    return hls_image

def bgr_to_gray(img):
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    gray = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)
    return gray


def gaussian_blur(img, ksize, sigma):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)

    pad_size = ksize // 2
    img_padded = np.pad(img, pad_size, mode='reflect')

    blurred_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            blurred_img[i, j] = np.sum(img_padded[i:i+ksize, j:j+ksize] * kernel)

    return blurred_img.astype(np.uint8)

def inrange(img, lower, upper):
    if(len(lower) == 3):
        lower = np.array(lower, dtype=img.dtype)
        upper = np.array(upper, dtype=img.dtype)
        mask = np.all((img >= lower) & (img <= upper), axis=-1).astype(np.uint8) * 255
        return mask

    else:
        mask = ((img >= lower[0]) & (img <= upper[0])).astype(np.uint8) * 255
        return mask
    
def count_nonzero(img):
    return np.sum(img > 0)
    

def sobel(img, dx, dy, ksize):
    if ksize == 3:
        kernel_x = np.array([[-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], 
                             [0, 0, 0], 
                             [1, 2, 1]])
    else:  # ksize == 5
        kernel_x = np.array([[-2, -1, 0, 1, 2],
                             [-3, -2, 0, 2, 3],
                             [-4, -3, 0, 3, 4],
                             [-3, -2, 0, 2, 3],
                             [-2, -1, 0, 1, 2]])
        kernel_y = kernel_x.T  

    kernel = kernel_x if dx == 1 else kernel_y
    pad_size = ksize // 2
    img_padded = np.pad(img, pad_size, mode='reflect')
    sobel_img = np.zeros_like(img, dtype=np.float64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sobel_img[i, j] = np.sum(img_padded[i:i+ksize, j:j+ksize] * kernel)

    return sobel_img


def line_intersection(line1, line2):
    """Check if two line segments intersect within the image bounds."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute determinants
    A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3

    det = A1 * B2 - A2 * B1 
    if abs(det) < 1e-6:
        return False 

    # Compute intersection point
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det

    def within_bounds(p, q, r):
        return min(p, q) <= r <= max(p, q)

    if within_bounds(x1, x2, x) and within_bounds(y1, y2, y) and \
       within_bounds(x3, x4, x) and within_bounds(y3, y4, y):
        return True 

    return False

def minline(x1,y1,x2,y2, minLineLength):
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_length > minLineLength:
        return True
    
def angle_with_horizontal(line):
    x1, y1, x2, y2 = line
    angle = abs(np.arctan2(y2 - y1, x2 - x1) * (180 / np.pi))
    return min(angle, 180 - angle)

def top(line, image_height):
    x1, y1, x2, y2 = line
    threshold_y = image_height * 0.3  # Top 30% threshold
    thr = image_height * 0.8
    return ( y1 < threshold_y and y2 < threshold_y)
def bottom(line, image_height):
    x1, y1, x2, y2 = line
    threshold_y = image_height * 0.75
    return (y1 > threshold_y and y2 > threshold_y)


def canny(img, low_threshold, high_threshold):
    sobelx = sobel(img, 1, 0, 3)
    sobely = sobel(img, 0, 1, 3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    edges = np.zeros_like(img, dtype=np.uint8)
    strong_edges = gradient_magnitude >= high_threshold
    weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold)
    edges[strong_edges] = 255
    edges[weak_edges] = 128

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] == 128 and np.any(edges[i-1:i+2, j-1:j+2] == 255):
                edges[i, j] = 255
            else:
                edges[i, j] = 0

    return edges

