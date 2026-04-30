# ==========================================================
# BLUE POINT DETECTOR
# ==========================================================
# Utility module for detecting and counting blue points
# in ear images (pre-piercing phase)
# ==========================================================

import cv2
import numpy as np
from typing import List, Tuple


def detect_blue_points(image: np.ndarray, min_area: int = 10) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Detect blue point markers in an image.
    
    Args:
        image: Input image (BGR format from OpenCV)
        min_area: Minimum contour area to consider as a valid point (default: 10 pixels)
    
    Returns:
        points: List of (x, y) coordinates of detected blue points (sorted top-to-bottom)
        mask: Binary mask showing detected blue regions
    """
    # Convert to HSV for better blue detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define blue color range in HSV
    # H: 90-130 (blue hue range)
    # S: 50-255 (saturation)
    # V: 50-255 (value/brightness)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create binary mask for blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours (blue dots)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by minimum area
        if area > min_area:
            # Calculate centroid
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                points.append((cx, cy))
    
    # Sort points from top to bottom (y-coordinate)
    points.sort(key=lambda p: p[1])
    
    return points, mask


def count_blue_points(image: np.ndarray, min_area: int = 10) -> int:
    """
    Count the number of blue points in an image.
    
    Args:
        image: Input image (BGR format from OpenCV)
        min_area: Minimum contour area to consider as a valid point
    
    Returns:
        count: Number of detected blue points
    """
    points, _ = detect_blue_points(image, min_area)
    return len(points)


def draw_detected_points(image: np.ndarray, points: List[Tuple[int, int]], 
                          point_radius: int = 6, point_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw detected points on an image with numbering.
    
    Args:
        image: Input image
        points: List of (x, y) coordinates
        point_radius: Radius of the circle marker
        point_color: BGR color tuple for the points
    
    Returns:
        Annotated image with drawn points and numbers
    """
    annotated = image.copy()
    
    for i, (x, y) in enumerate(points):
        # Draw circle around the point
        cv2.circle(annotated, (x, y), point_radius, point_color, -1)
        
        # Draw white outline
        cv2.circle(annotated, (x, y), point_radius, (255, 255, 255), 1)
        
        # Draw point number
        cv2.putText(
            annotated,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
    
    return annotated
