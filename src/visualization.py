"""
Visualization Tools for Motion Estimation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_flow(frame, flow, step=16, color=(0, 255, 0), thickness=1):
    """
    Draw optical flow vectors on frame
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame (BGR)
    flow : np.ndarray
        Optical flow field
    step : int
        Spacing between flow vectors
    color : tuple
        Color of vectors (BGR)
    thickness : int
        Thickness of vectors
    
    Returns:
    --------
    result : np.ndarray
        Frame with flow vectors drawn
    """
    result = frame.copy()
    h, w = flow.shape[:2]
    
    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    for (x1, y1), (x2, y2) in lines:
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(result, (x2, y2), 3, color, -1)
    
    return result


def draw_trajectory(frame, trajectory, color=(0, 255, 0), thickness=2, radius=5):
    """
    Draw trajectory on frame
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame (BGR)
    trajectory : list of tuples
        Sequence of (x, y) coordinates
    color : tuple
        Color (BGR)
    thickness : int
        Line thickness
    radius : int
        Circle radius for keypoints
    
    Returns:
    --------
    result : np.ndarray
        Frame with trajectory drawn
    """
    result = frame.copy()
    
    if len(trajectory) > 1:
        # Draw trajectory line
        pts = np.array(trajectory, dtype=np.int32)
        cv2.polylines(result, [pts], False, color, thickness)
    
    # Draw start and end points
    if trajectory:
        start = tuple(map(int, trajectory[0]))
        end = tuple(map(int, trajectory[-1]))
        
        cv2.circle(result, start, radius, (0, 255, 0), -1)  # Green start
        cv2.circle(result, end, radius, (0, 0, 255), -1)    # Red end
        
        # Draw intermediate keypoints
        for i, point in enumerate(trajectory[1:-1], 1):
            pt = tuple(map(int, point))
            cv2.circle(result, pt, radius // 2, (255, 0, 0), -1)  # Blue intermediate
    
    return result


def plot_flow_magnitude(flow, title="Optical Flow Magnitude"):
    """
    Plot optical flow magnitude as heatmap
    
    Parameters:
    -----------
    flow : np.ndarray
        Optical flow field
    title : str
        Plot title
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap='hot')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


def plot_trajectory_analysis(trajectory, stats=None):
    """
    Plot trajectory analysis
    
    Parameters:
    -----------
    trajectory : list of tuples
        Sequence of (x, y) coordinates
    stats : dict
        Trajectory statistics
    """
    trajectory = np.array(trajectory)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trajectory path
    ax = axes[0, 0]
    if trajectory.size == 0:
        ax.text(0.5, 0.5, 'No trajectory points available', ha='center', va='center')
        ax.set_title('Object Trajectory')
        ax.axis('off')
    elif len(trajectory) == 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'bo', markersize=10)
        ax.set_title('Object Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True)
        ax.axis('equal')
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        ax.set_title('Object Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
    
    # Plot 2: Velocity over time
    ax = axes[0, 1]
    if len(trajectory) > 1:
        displacements = np.diff(trajectory, axis=0)
        velocities = np.linalg.norm(displacements, axis=1)
        ax.plot(velocities, 'r-', linewidth=2)
        ax.set_title('Velocity Over Time')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Velocity (pixels/frame)')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'Insufficient data to compute velocity', ha='center', va='center')
        ax.axis('off')
        displacements = None
    
    # Plot 3: Direction over time
    ax = axes[1, 0]
    if len(trajectory) > 1:
        angles = np.arctan2(displacements[:, 1], displacements[:, 0]) * 180 / np.pi
        ax.plot(angles, 'g-', linewidth=2)
        ax.set_title('Direction Over Time')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'Insufficient data to compute direction', ha='center', va='center')
        ax.axis('off')
    
    # Plot 4: Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    if stats:
        text = "Trajectory Statistics:\n\n"
        for key, value in stats.items():
            if isinstance(value, tuple):
                text += f"{key}: ({value[0]:.2f}, {value[1]:.2f})\n"
            elif isinstance(value, float):
                text += f"{key}: {value:.4f}\n"
            else:
                text += f"{key}: {value}\n"
        ax.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def save_visualization(output_path, fig):
    """Save matplotlib figure"""
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
