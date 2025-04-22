import numpy as np
import cv2

def fit_curve(edges, degree=2):
    x, y = edges
    fit_params = np.polyfit(y, x, degree) 
    return fit_params

def get_polynomial_points(fit_params, y_values):
    x_values = np.polyval(fit_params, y_values)  
    return x_values

def detect_lane(edges, y_range):
    x, y = edges
    midpoint = np.mean(x)
    left_edges = (x[x < midpoint], y[x < midpoint])
    right_edges = (x[x >= midpoint], y[x >= midpoint])

    fit_params_left = fit_curve(left_edges, degree=2)
    fit_params_right = fit_curve(right_edges, degree=2)

    y_values = np.linspace(y_range[0], y_range[1], 100)
    x_left = get_polynomial_points(fit_params_left, y_values)
    x_right = get_polynomial_points(fit_params_right, y_values)

    left_line_points = np.vstack((x_left, y_values)).T
    right_line_points = np.vstack((x_right, y_values)).T

    return left_line_points, right_line_points

def draw(image, left_line_points, right_line_points, color=(0, 255, 0), thickness=5):
    left_line_points = np.int32(left_line_points)
    right_line_points = np.int32(right_line_points)
    for i in range(len(left_line_points) - 1):
        cv2.line(image, tuple(left_line_points[i]), tuple(left_line_points[i + 1]), color, thickness)
        cv2.line(image, tuple(right_line_points[i]), tuple(right_line_points[i + 1]), color, thickness)
    return image
