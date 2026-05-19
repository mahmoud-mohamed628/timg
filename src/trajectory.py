"""
Trajectory Extraction and Analysis
"""

import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment


def extract_trajectory_from_flow(flow_sequence, threshold=0.5):
    """
    Extract object trajectory from optical flow sequence
    
    Parameters:
    -----------
    flow_sequence : list of np.ndarray
        Sequence of optical flow fields
    threshold : float
        Motion threshold for detecting the object
    
    Returns:
    --------
    trajectory : list of tuples
        Sequence of (x, y) coordinates
    """
    trajectory = []
    
    for flow in flow_sequence:
        # Compute motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Threshold to find moving regions
        motion_mask = magnitude > threshold
        
        if motion_mask.sum() > 0:
            # Find center of mass of moving region
            labeled, num_features = label(motion_mask)
            centers = center_of_mass(motion_mask, labeled, range(1, num_features + 1))
            
            if centers:
                # Use largest moving region
                sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
                largest_idx = np.argmax(sizes)
                cy, cx = centers[largest_idx]
                trajectory.append((cx, cy))
    
    return trajectory


def analyze_trajectory(trajectory):
    """
    Analyze trajectory statistics
    
    Parameters:
    -----------
    trajectory : list of tuples
        Sequence of (x, y) coordinates
    
    Returns:
    --------
    stats : dict
        Trajectory statistics (velocity, direction, etc.)
    """
    trajectory = np.array(trajectory)
    
    stats = {
        'trajectory_length': len(trajectory),
        'start_position': tuple(trajectory[0]) if len(trajectory) >= 1 else None,
        'end_position': tuple(trajectory[-1]) if len(trajectory) >= 1 else None,
    }

    if len(trajectory) < 2:
        stats['error'] = 'Trajectory too short'
        return stats
    
    # Compute displacement vectors
    displacements = np.diff(trajectory, axis=0)
    
    # Compute velocity (magnitude)
    velocities = np.linalg.norm(displacements, axis=1)
    
    # Compute direction angles
    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    
    stats.update({
        'total_distance': np.sum(velocities),
        'avg_velocity': np.mean(velocities),
        'max_velocity': np.max(velocities),
        'min_velocity': np.min(velocities),
        'avg_direction': np.mean(angles),
    })
    
    return stats


def track_object_centroid(frames, method='contour', min_area=100):
    """
    Track object centroid across frames
    
    Parameters:
    -----------
    frames : list of np.ndarray
        Sequence of color frames
    method : str
        'contour' or 'background_subtraction'
    min_area : int
        Minimum contour area to consider
    
    Returns:
    --------
    trajectory : list of tuples
        Sequence of (x, y) coordinates
    """
    trajectory = []
    
    if method == 'background_subtraction':
        # Use MOG2 background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2()
        
        for frame in frames:
            mask = backSub.apply(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > min_area:
                    M = cv2.moments(largest_contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        trajectory.append((cx, cy))
    
    elif method == 'contour':
        # Use simple edge detection
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > min_area:
                    M = cv2.moments(largest_contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        trajectory.append((cx, cy))
            
            prev_gray = gray
    
    return trajectory
