import cv2
import numpy as np

def hough_transformation(image, output_name):
    object_edges = cv2.Canny(image, 10, 200)
    edge_points = np.nonzero(object_edges)

    
    