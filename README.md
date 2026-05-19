# Motion Estimation for a Single Object

## Objective
Estimate the motion of a single object in an image sequence and analyze its trajectory using the Lucas-Kanade optical flow method.

## Project Overview
This project implements motion estimation techniques to:
- Estimate the motion field of an object using Lucas-Kanade sparse optical flow
- Extract and visualize the global trajectory
- Analyze velocity and direction of movement over time
- Generate comprehensive graphs and reports

## Features
- Interactive video selection via file dialog
- Lucas-Kanade optical flow estimation
- Real-time motion visualization with arrows and circles
- Trajectory extraction and analysis
- Automatic graph generation (trajectory path, velocity, direction)
- Dynamic report naming based on video filename

## Project Structure
```
prjSujet_1/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                 # Main motion estimation script
├── rapports_graphes/       # Generated reports and graphs directory
└── vedios/                 # Input video files directory
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

The script will:
1. Open a file dialog to select a video file
2. Extract features using Shi-Tomasi corner detection
3. Track features using Lucas-Kanade optical flow
4. Display real-time motion visualization
5. Generate trajectory path, velocity, and direction graphs
6. Save the analysis plots to `rapports_graphes/` folder

### Controls
- Press `q` to stop video processing early (optional)

## Supported Video Formats
- MP4
- AVI
- MOV
- MKV
- Other formats supported by OpenCV

## Output
Generated graphs include:
1. **Trajectory Path**: 2D plot of object position over time
2. **Velocity Graph**: Speed (pixels/frame) over time
3. **Direction Graph**: Movement direction (degrees) over time

All outputs are saved in the `rapports_graphes/` directory with the video name as prefix.

## Technical Details

### Lucas-Kanade Method
- Local motion estimation technique
- Tracks feature points across consecutive frames
- Window-based approach using image pyramids
- Parameters:
  - Window size: 15×15 pixels
  - Max pyramid levels: 2
  - Feature detection quality level: 0.01

### Shi-Tomasi Corner Detection
- Identifies strong feature points in the first frame
- Maximum 50 corners tracked
- Minimum distance between features: 10 pixels

## Deadline
**May 19, 2026 - 22:00**

## References
- Lucas & Kanade (1981) - "An Iterative Image Registration Technique"
- Shi & Tomasi (1994) - "Good Features to Track"
- OpenCV Documentation on Optical Flow

## License
Academic Project
