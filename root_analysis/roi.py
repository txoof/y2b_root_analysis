# library/roi.py
"""Functions for detecting and working with regions of interest (ROI) in images."""

from pathlib import Path
import cv2
import numpy as np


def ensure_grayscale(im):
    """Convert an image to grayscale if it is not already.
    
    Args:
        im: Input image array. Can be either a 2D grayscale image or a 3D color 
            image with 3 channels (BGR format).
    
    Returns:
        Grayscale image as a 2D array.
    
    Raises:
        ValueError: If the input image has an unexpected shape (neither 2D nor 3D 
            with 3 channels).
    """
    if len(im.shape) == 3 and im.shape[2] == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif len(im.shape) == 2:
        gray = im
    else:
        raise ValueError(f"Unexpected image shape: {im.shape}")
    
    return gray


def get_binary_mask(image, thresh=128, maxval=255):
    """Create a binary mask using Otsu's thresholding.
    
    Args:
        image: Input image (can be color or grayscale).
        thresh: Initial threshold value (ignored when using Otsu's method). 
            Default is 128.
        maxval: Value to assign to pixels greater than the threshold. Default is 255.
    
    Returns:
        Tuple of (threshold_value, binary_mask) where threshold_value is the computed 
        threshold from Otsu's method and binary_mask is a uint8 array with values 
        in {0, maxval}.
    """
    gray = ensure_grayscale(image)
    ret, binary_mask = cv2.threshold(
        gray, thresh=thresh, maxval=maxval, 
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return ret, binary_mask


def get_bounding_box(binary_mask):
    """Compute a square bounding box around the largest contour in a binary mask.
    
    Args:
        binary_mask: 2D binary image (dtype uint8) where foreground pixels are 
            non-zero (e.g. 255).
    
    Returns:
        Tuple of ((x1, y1), (x2, y2)) giving the top-left and bottom-right 
        coordinates of a square bounding box that encloses the largest contour. 
        Coordinates are clamped to the image boundaries.
    
    Raises:
        ValueError: If no contours are found in the provided mask.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        raise ValueError("No contours found in binary mask")
    
    # Get the biggest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the x, y, width, height of the contour
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Get the longest side
    size = max(w, h)
    
    # Find the center of each side
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate the new x, y based on the longest side
    new_x = center_x - size // 2
    new_y = center_y - size // 2
    
    # Clamp to image boundaries
    new_x = max(0, min(new_x, binary_mask.shape[1] - size))
    new_y = max(0, min(new_y, binary_mask.shape[0] - size))
    
    return ((new_x, new_y), (new_x + size, new_y + size))


def detect_roi(image):
    """Detect the ROI (e.g., petri dish) bounding box in an image.
    
    Args:
        image: Input image (BGR color or grayscale).
    
    Returns:
        ROI bounding box as ((x1, y1), (x2, y2)).
    """
    _, binary_mask = get_binary_mask(image)
    roi_bbox = get_bounding_box(binary_mask)
    return roi_bbox

def crop_to_roi(image, roi_bbox):
    """Crop an image to its ROI bounding box.
    
    Args:
        image: Input image (can be color or grayscale).
        roi_bbox: ((x1, y1), (x2, y2)) from detect_roi.
    
    Returns:
        Cropped image.
    """
    (x1, y1), (x2, y2) = roi_bbox
    return image[y1:y2, x1:x2]

