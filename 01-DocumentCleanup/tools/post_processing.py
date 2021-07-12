import cv2
import numpy as np

from scipy.optimize import minimize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def binarize_image(image, remove_noise=False):
    image = np.array(image * 255, dtype=np.uint8)
    if remove_noise:
        image = cv2.medianBlur(image, 9)

    _, bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bin_image


def get_corner_points(points, shape):
    references = np.array([
        [0, 0],
        [shape[1], 0],
        [0, shape[0]]
    ])
    corner_points = np.zeros((3, 2))
    best_dist = np.ones(3) * 1000
    points = np.array(points)

    for i, ref in enumerate(references):
        for p in points:
            dist = np.linalg.norm(p - ref)
            if dist < best_dist[i]:
                best_dist[i] = dist
                corner_points[i] = p

    # resize
    avg_point = np.mean(points, axis=0)
    tmp_points = (corner_points - avg_point) * 1.25
    corner_points = tmp_points + avg_point
        
    return corner_points
 

def get_approx_polygon(image, alpha=0.009):
    """Calcualte the approximate polygon based on the object contour
    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [ cv2.contourArea(cnt) for cnt in contours ]
    contour = contours[np.argmax(areas)]
    approx = cv2.approxPolyDP(contour, alpha * cv2.arcLength(contour, True), True)
    return approx


def rectify_image(image):
    """Method responsible to rectify and binarize the input image
    """

    # Get a smoothed binary image
    bin_image = binarize_image(image, remove_noise=True)

    # Apply morphologycal operations to merge nearby letters
    kernel_size = int(np.mean(bin_image.shape) * 0.07)
    kernel = np.ones((kernel_size, ) * 2 )
    image_opened = cv2.morphologyEx(255 - bin_image, cv2.MORPH_CLOSE, kernel)
    
    # Get the approximated polygon (something similar to a quadrilateral)
    approx = get_approx_polygon(image_opened)
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get corner points used to make affine transforms
    src_tri_1 = get_corner_points(box, image.shape).astype(np.float32)
    src_tri_2 = get_corner_points(np.squeeze(approx, axis=1), image.shape).astype(np.float32)
    src_tri = (src_tri_1 + src_tri_2) / 2
    dst_tri = np.array([
        [0., 0.],
        [image.shape[1], 0],
        [0, image.shape[0]],
    ]).astype(np.float32)

    # Rectify final image mask
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    image = binarize_image(image)
    image = cv2.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))

    return image