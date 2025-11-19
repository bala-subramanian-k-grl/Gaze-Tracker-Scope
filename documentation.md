# Gaze Tracker System - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Technical Specifications](#technical-specifications)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [Calibration System](#calibration-system)
9. [Data Format & Storage](#data-format--storage)
10. [API Reference](#api-reference)
11. [Algorithm Details](#algorithm-details)
12. [Performance & Optimization](#performance--optimization)
13. [Troubleshooting](#troubleshooting)
14. [Future Enhancements](#future-enhancements)

---

## Overview

The **Gaze Tracker System** is a real-time, hybrid gaze tracking solution that combines head pose estimation and eye gaze analysis to accurately determine where a user is looking on a screen. Built using OpenCV and MediaPipe, it provides robust gaze tracking capabilities suitable for research, accessibility applications, and human-computer interaction systems.

### Key Features

- **Real-Time Processing**: Processes video frames at 30 FPS with low latency
- **Hybrid Gaze Estimation**: Combines head pose and eye gaze for improved accuracy
- **Dual Calibration Modes**: Point-based manual calibration and automatic adaptive calibration
- **Robust Tracking**: Handles head movements, blinks, and varying lighting conditions
- **Comprehensive Data Logging**: Exports gaze data in JSON format for analysis
- **Visual Feedback**: Real-time visualization with calibration indicators
- **Configurable Parameters**: Extensive configuration options for different use cases

### Use Cases

- **Accessibility**: Eye-controlled interfaces for users with limited mobility
- **Research**: Eye movement studies and attention analysis
- **User Experience**: Gaze-based interaction and heatmap generation
- **Gaming**: Eye-tracking enhanced gameplay
- **Medical Applications**: Vision therapy and rehabilitation

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gaze Tracker System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Camera     │─────▶│  MediaPipe   │                     │
│  │   Input      │      │  Face Mesh   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                    ┌──────────┴──────────┐                  │
│                    │                     │                   │
│            ┌───────▼──────┐      ┌───────▼──────┐           │
│            │ Head Pose    │      │ Eye Gaze    │           │
│            │ Estimator    │      │ Estimator   │           │
│            └───────┬──────┘      └───────┬──────┘           │
│                    │                     │                   │
│                    └──────────┬──────────┘                  │
│                               │                              │
│                    ┌──────────▼──────────┐                  │
│                    │  Hybrid Fusion      │                  │
│                    │  Engine             │                  │
│                    └──────────┬──────────┘                  │
│                               │                              │
│                    ┌──────────▼──────────┐                  │
│                    │  Calibration        │                  │
│                    │  System             │                  │
│                    └──────────┬──────────┘                  │
│                               │                              │
│                    ┌──────────▼──────────┐                  │
│                    │  Smoothing Filter   │                  │
│                    └──────────┬──────────┘                  │
│                               │                              │
│                    ┌──────────▼──────────┐                  │
│                    │  Output &           │                  │
│                    │  Visualization      │                  │
│                    └─────────────────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Input Layer**: Camera captures RGB frames
2. **Detection Layer**: MediaPipe detects facial landmarks and iris positions
3. **Estimation Layer**: Parallel processing of head pose and eye gaze
4. **Fusion Layer**: Weighted combination of head and eye gaze vectors
5. **Calibration Layer**: Applies calibration model if available
6. **Smoothing Layer**: Reduces jitter and noise
7. **Output Layer**: Visualizes results and logs data

---

## Core Components

### 1. GazeConfig Class

Central configuration class that manages all system parameters.

**Location**: `Gaze_Tracker.py` (lines 31-72)

**Key Parameters**:
- **Camera Settings**: ID, resolution (1280x720), FPS (30)
- **MediaPipe Settings**: Max faces, landmark refinement, confidence thresholds
- **Landmark Indices**: Face points, eye regions, iris positions
- **Smoothing Factors**: Eye gaze (0.7) and head pose (0.8)
- **Fusion Weights**: Eye gaze (70%), head pose (30%)
- **Calibration Settings**: 9-point grid, duration per point (3s)

### 2. HeadPoseEstimator Class

Estimates 3D head orientation using solvePnP algorithm.

**Location**: `Gaze_Tracker.py` (lines 74-179)

**Key Methods**:
- `estimate_pose(landmarks)`: Main pose estimation using 6 facial landmarks
- `rotation_matrix_to_euler_angles(R)`: Converts rotation matrix to pitch/yaw/roll
- `head_pose_to_gaze_direction(rotation_angles, translation_vec)`: Maps head pose to gaze coordinates

**Algorithm**:
1. Extracts 2D image points from selected facial landmarks
2. Uses solvePnP with 3D model points to estimate pose
3. Converts rotation vector to Euler angles
4. Applies exponential smoothing
5. Maps angles to normalized gaze coordinates (0-1)

**3D Model Points** (in mm):
- Nose tip: (0, 0, 0)
- Left eye corner: (-225, 170, -135)
- Right eye corner: (225, 170, -135)
- Left mouth: (-150, -150, -125)
- Right mouth: (150, -150, -125)
- Chin: (0, -330, -65)

### 3. GazeCalibrator Class

Manages calibration data collection and model training.

**Location**: `Gaze_Tracker.py` (lines 181-387)

**Calibration Modes**:
- **Point Calibration**: User looks at 9 predefined points on screen
- **Automatic Calibration**: Continuous learning during normal use

**Key Methods**:
- `start_point_calibration()`: Initiates 9-point calibration procedure
- `start_automatic_calibration()`: Begins automatic calibration mode
- `add_calibration_sample()`: Records calibration data points
- `compute_calibration_model()`: Trains polynomial regression model
- `apply_calibration()`: Applies trained model to raw gaze coordinates
- `save_calibration()` / `load_calibration()`: Persistence operations

**Calibration Model**:
- Uses polynomial regression (degree 2) from scikit-learn
- Maps raw gaze coordinates to screen coordinates
- Requires minimum 4 calibration points
- Saves model coefficients for later use

### 4. HybridGazeTracker Class

Main system class that orchestrates all components.

**Location**: `Gaze_Tracker.py` (lines 389-869)

**Key Methods**:
- `process_frame(frame)`: Main processing pipeline for each frame
- `estimate_eye_gaze_direction()`: Calculates gaze from iris position
- `combine_gaze_estimates()`: Fuses head and eye gaze with adaptive weighting
- `smooth_gaze_coordinates()`: Applies exponential smoothing
- `draw_enhanced_visualization()`: Renders tracking overlay

**Processing Pipeline**:
1. Convert frame to RGB
2. Process with MediaPipe Face Mesh
3. Extract landmarks for head pose and eyes
4. Estimate head pose (pitch, yaw, roll)
5. Calculate eye gaze from iris position
6. Combine estimates with weighted fusion
7. Apply calibration if available
8. Smooth coordinates
9. Visualize and log data

---

## Technical Specifications

### System Requirements

**Hardware**:
- Webcam or USB camera (minimum 640x480, recommended 1280x720)
- CPU: Multi-core processor (Intel i5 or equivalent)
- RAM: 4GB minimum, 8GB recommended
- Operating System: Windows, macOS, or Linux

**Software**:
- Python 3.7 or higher
- OpenCV 4.8.0+
- MediaPipe 0.10.0+
- NumPy 1.21.0+
- scikit-learn 1.0.0+ (for calibration)

### Performance Metrics

- **Frame Rate**: 30 FPS (configurable)
- **Latency**: ~33ms per frame
- **Accuracy**: 
  - Uncalibrated: ±5-10% screen space
  - Calibrated: ±2-5% screen space
- **CPU Usage**: 15-30% on modern CPUs
- **Memory Usage**: ~200-500 MB

### Supported Camera Resolutions

- 640x480 (VGA)
- 1280x720 (HD) - Recommended
- 1920x1080 (Full HD) - Supported but may reduce FPS

---

## Installation & Setup

### Step 1: Prerequisites

Ensure Python 3.7+ is installed:
```bash
python --version
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd Gaze-Tracker-Scope
```

### Step 3: Create Virtual Environment (Recommended)

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages**:
- `opencv-python>=4.8.0`
- `mediapipe>=0.10.0`
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `matplotlib>=3.5.0`
- `scikit-learn>=1.0.0`

### Step 5: Verify Installation

```bash
python -c "import cv2, mediapipe, numpy; print('All packages installed successfully')"
```

### Step 6: Run the System

```bash
python Gaze_Tracker.py
```

---

## Configuration

### GazeConfig Parameters

All configuration is managed through the `GazeConfig` class. Modify these values in `Gaze_Tracker.py`:

#### Camera Configuration

```python
CAMERA_ID = 0                    # Camera device index
CAMERA_WIDTH = 1280              # Frame width in pixels
CAMERA_HEIGHT = 720              # Frame height in pixels
FPS = 30                         # Target frames per second
```

#### MediaPipe Settings

```python
MAX_NUM_FACES = 1                # Maximum faces to detect
REFINE_LANDMARKS = True          # Enable refined landmarks (includes iris)
MIN_DETECTION_CONFIDENCE = 0.7   # Minimum confidence for face detection
MIN_TRACKING_CONFIDENCE = 0.5    # Minimum confidence for tracking
```

#### Smoothing Parameters

```python
SMOOTHING_FACTOR = 0.7           # Eye gaze smoothing (0-1, higher = more smoothing)
HEAD_SMOOTHING_FACTOR = 0.8      # Head pose smoothing (0-1)
```

#### Fusion Weights

```python
EYE_GAZE_WEIGHT = 0.7            # Weight for eye gaze (0-1)
HEAD_GAZE_WEIGHT = 0.3           # Weight for head pose (0-1)
MIN_CONFIDENCE_THRESHOLD = 0.4   # Minimum confidence to apply smoothing
```

#### Calibration Settings

```python
CALIBRATION_POINTS = [           # 9-point calibration grid (normalized 0-1)
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
]
CALIBRATION_DURATION = 3.0        # Seconds per calibration point
CALIBRATION_SAMPLES_PER_POINT = 30  # Target samples per point
```

### Advanced Configuration

#### Adjusting for Different Users

For users with glasses or different eye shapes:
- Increase `MIN_DETECTION_CONFIDENCE` to 0.8
- Decrease `SMOOTHING_FACTOR` to 0.5 for faster response
- Adjust `EYE_GAZE_WEIGHT` based on eye visibility

#### Performance Tuning

For lower-end hardware:
- Reduce `CAMERA_WIDTH` to 640 and `CAMERA_HEIGHT` to 480
- Set `REFINE_LANDMARKS = False` (disables iris detection)
- Increase `SMOOTHING_FACTOR` to 0.9 to reduce processing

---

## Usage Guide

### Basic Usage

1. **Start the Application**:
   ```bash
   python Gaze_Tracker.py
   ```

2. **Position Yourself**: Sit 50-70cm from the camera, ensure good lighting

3. **Calibration** (Recommended):
   - Press `c` to start point calibration
   - Look at each of the 9 points as they appear
   - Wait 3 seconds per point
   - System will automatically compute calibration model

4. **Tracking**: Gaze point will be displayed as a colored circle

5. **Save Data**: Press `s` to save current session data

6. **Quit**: Press `q` to exit

### Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| `q` | Quit | Exit the application |
| `s` | Save Data | Export current gaze data to JSON file |
| `c` | Point Calibration | Start 9-point calibration procedure |
| `a` | Auto Calibration | Start automatic calibration mode |
| `l` | Load Calibration | Load previously saved calibration |
| `r` | Reset Calibration | Clear current calibration |

### Calibration Procedure

#### Point Calibration (Recommended)

1. Press `c` to start calibration
2. A grid of 9 points will appear on screen
3. The current point will pulse with yellow circle
4. Look directly at the pulsing point
5. Hold your gaze for 3 seconds
6. System automatically moves to next point
7. After all 9 points, calibration model is computed
8. Calibration file is saved automatically

**Tips for Best Results**:
- Keep head still during calibration
- Look directly at the center of each point
- Ensure good lighting
- Remove glasses if possible (or calibrate with glasses)

#### Automatic Calibration

1. Press `a` to start automatic calibration
2. System collects data during normal use
3. Requires at least 20 samples
4. Less accurate than point calibration but more convenient

### Data Collection

Gaze data is automatically logged during operation. To save:

1. Press `s` during operation
2. Data is saved to `hybrid_gaze_data_YYYYMMDD_HHMMSS.json`
3. Includes all gaze coordinates, confidence, and metadata

---

## Calibration System

### Overview

The calibration system maps raw gaze coordinates (derived from facial features) to actual screen coordinates. This compensates for individual differences in eye geometry, head position, and camera setup.

### Calibration Modes

#### 1. Point Calibration

**Method**: User looks at 9 predefined points on screen

**Process**:
1. System displays 9 points in a 3x3 grid
2. User looks at each point for 3 seconds
3. System collects ~30 samples per point
4. Averages samples for each point
5. Trains polynomial regression model (degree 2)
6. Validates model with Mean Squared Error (MSE)

**Advantages**:
- High accuracy (±2-5% screen space)
- Fast calibration (~30 seconds)
- Reliable and reproducible

**Disadvantages**:
- Requires user cooperation
- May be tedious for some users

#### 2. Automatic Calibration

**Method**: Continuous learning during normal use

**Process**:
1. System collects gaze data during operation
2. Uses clustering or regression techniques
3. Adapts model over time
4. Requires minimum 20 samples

**Advantages**:
- No explicit calibration needed
- Adapts to user over time

**Disadvantages**:
- Lower initial accuracy
- Requires more data
- May drift over time

### Calibration Model

**Type**: Polynomial Regression (degree 2)

**Input**: Raw gaze coordinates (x, y) normalized 0-1

**Output**: Calibrated screen coordinates (x, y) normalized 0-1

**Model Structure**:
```
calibrated_x = f(raw_x, raw_y) = a₀ + a₁x + a₂y + a₃x² + a₄xy + a₅y²
calibrated_y = g(raw_x, raw_y) = b₀ + b₁x + b₂y + b₃x² + b₄xy + b₅y²
```

**Coefficients**: Stored in calibration JSON file for persistence

### Calibration File Format

Calibration files are saved as JSON with the following structure:

```json
{
  "timestamp": "2025-11-19T14:59:10.553648",
  "is_calibrated": true,
  "calibration_mode": "POINT_CALIBRATION",
  "model_coefficients": {
    "coefficients": [[...], [...]],
    "intercept": [...],
    "poly_features": [[...], [...]]
  },
  "calibration_samples_summary": {
    "0": 87,
    "1": 81,
    ...
  }
}
```

### Calibration Quality Metrics

The system reports:
- **MSE (Mean Squared Error)**: Lower is better (typically < 0.01)
- **Sample Count**: Number of samples per calibration point
- **Coverage**: Number of calibration points successfully collected

### Recalibration

Recalibration is recommended when:
- User changes position relative to screen
- Camera is moved
- Lighting conditions change significantly
- Accuracy degrades over time

---

## Data Format & Storage

### Gaze Data Structure

Each gaze data point contains:

```json
{
  "timestamp": "ISO 8601 timestamp",
  "gaze_x": 0.0-1.0,              // Normalized X coordinate
  "gaze_y": 0.0-1.0,              // Normalized Y coordinate
  "confidence": 0.0-1.0,          // Overall confidence score
  "fps": 30.0,                    // Current frame rate
  "calibrated": true/false,       // Whether calibration is applied
  "eye_gaze_x": 0.0-1.0,          // Eye-only gaze X
  "eye_gaze_y": 0.0-1.0,          // Eye-only gaze Y
  "head_gaze_x": 0.0-1.0,         // Head-only gaze X
  "head_gaze_y": 0.0-1.0,         // Head-only gaze Y
  "raw_gaze_x": 0.0-1.0,          // Raw gaze X (before calibration)
  "raw_gaze_y": 0.0-1.0,          // Raw gaze Y (before calibration)
  "eye_confidence": 0.0-1.0,      // Eye detection confidence
  "eye_closure_ratio": 0.0-2.0,   // Eye aspect ratio (blink detection)
  "eye_weight": 0.0-1.0,          // Weight used for eye gaze in fusion
  "head_weight": 0.0-1.0,         // Weight used for head pose in fusion
  "head_pose_angles": [           // Head pose Euler angles (radians)
    pitch,                        // Rotation around X-axis
    yaw,                          // Rotation around Y-axis
    roll                          // Rotation around Z-axis
  ],
  "resolution": "1280x720"        // Camera resolution
}
```

### File Naming Convention

- **Gaze Data**: `hybrid_gaze_data_YYYYMMDD_HHMMSS.json`
- **Calibration**: `gaze_calibration_YYYYMMDD_HHMMSS.json`

### Data Export

Data is saved as JSON array of gaze data points. To convert to CSV:

```python
import json
import pandas as pd

with open('hybrid_gaze_data_20251119_150556.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv('gaze_data.csv', index=False)
```

### Data Analysis

Common analyses:
- **Gaze Heatmaps**: Visualize where user looked most
- **Fixation Detection**: Identify stable gaze points
- **Saccade Analysis**: Detect rapid eye movements
- **Attention Patterns**: Analyze gaze distribution over time

---

## API Reference

### HybridGazeTracker Class

#### `__init__(config: GazeConfig)`

Initialize the gaze tracker with configuration.

**Parameters**:
- `config`: GazeConfig instance with system parameters

**Returns**: None

#### `process_frame(frame: np.ndarray) -> Tuple[np.ndarray, float, float, float, Dict]`

Process a single video frame for gaze detection.

**Parameters**:
- `frame`: BGR image frame from camera

**Returns**:
- `processed_frame`: Annotated frame with visualization
- `gaze_x`: Normalized X coordinate (0-1)
- `gaze_y`: Normalized Y coordinate (0-1)
- `confidence`: Confidence score (0-1)
- `debug_info`: Dictionary with detailed tracking information

#### `start_calibration(mode: str = "point")`

Start calibration procedure.

**Parameters**:
- `mode`: "point" for point calibration, "auto" for automatic

**Returns**: None

#### `get_gaze_data(gaze_x, gaze_y, confidence, debug_info) -> Dict`

Format gaze data for output.

**Parameters**:
- `gaze_x`, `gaze_y`: Gaze coordinates
- `confidence`: Confidence score
- `debug_info`: Debug information dictionary

**Returns**: Formatted gaze data dictionary

### HeadPoseEstimator Class

#### `estimate_pose(landmarks) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

Estimate head pose from facial landmarks.

**Parameters**:
- `landmarks`: MediaPipe landmark list

**Returns**:
- `rotation_angles`: Euler angles (pitch, yaw, roll) in radians
- `translation_vec`: Translation vector
- `rotation_mat`: 3x3 rotation matrix
- `translation_vec_3d`: 3D translation vector

### GazeCalibrator Class

#### `apply_calibration(raw_gaze_x: float, raw_gaze_y: float) -> Tuple[float, float]`

Apply calibration model to raw gaze coordinates.

**Parameters**:
- `raw_gaze_x`, `raw_gaze_y`: Raw gaze coordinates (0-1)

**Returns**: Calibrated coordinates (0-1)

#### `save_calibration(filename: str = None)`

Save calibration data to file.

**Parameters**:
- `filename`: Optional filename, defaults to timestamped name

**Returns**: None

#### `load_calibration(filename: str) -> bool`

Load calibration from file.

**Parameters**:
- `filename`: Path to calibration JSON file

**Returns**: True if loaded successfully, False otherwise

---

## Algorithm Details

### Head Pose Estimation

**Algorithm**: Perspective-n-Point (PnP) Problem

**Steps**:
1. Extract 6 facial landmark points (2D image coordinates)
2. Map to known 3D model points (in millimeters)
3. Solve PnP using `cv2.solvePnP()` with iterative method
4. Convert rotation vector to rotation matrix using Rodrigues formula
5. Extract Euler angles (pitch, yaw, roll) from rotation matrix
6. Apply exponential smoothing to reduce noise
7. Map Euler angles to normalized gaze coordinates

**Coordinate System**:
- X-axis: Left-right (positive = right)
- Y-axis: Up-down (positive = down)
- Z-axis: Forward-backward (positive = forward)

**Gaze Mapping**:
```python
head_gaze_x = 0.5 + (yaw / π) * 0.8
head_gaze_y = 0.5 + (pitch / π) * 0.8
```

### Eye Gaze Estimation

**Algorithm**: Iris Position Relative to Eye Region

**Steps**:
1. Extract eye region landmarks (16 points per eye)
2. Calculate eye bounding box
3. Extract iris center from 5 iris landmarks
4. Normalize iris position within eye region
5. Apply sigmoid transformation for non-linear mapping
6. Calculate confidence based on eye size

**Iris Normalization**:
```python
raw_gaze_x = (iris_x - eye_left) / eye_width
raw_gaze_y = (iris_y - eye_top) / eye_height
```

**Sigmoid Transformation**:
```python
gaze_x = 1 / (1 + exp(-8 * (raw_gaze_x - 0.5)))
gaze_y = 1 / (1 + exp(-8 * (raw_gaze_y - 0.5)))
```

### Hybrid Fusion

**Algorithm**: Weighted Combination with Adaptive Weights

**Base Weights**:
- Eye gaze: 70%
- Head pose: 30%

**Adaptive Adjustment**:
- If eye closed (closure ratio < 0.2): Eye 10%, Head 90%
- If eye confidence low: Reduce eye weight proportionally
- Normalize weights to sum to 1.0

**Combination**:
```python
combined_x = eye_gaze_x * eye_weight + head_gaze_x * head_weight
combined_y = eye_gaze_y * eye_weight + head_gaze_y * head_weight
```

### Smoothing Filter

**Algorithm**: Exponential Moving Average

**Formula**:
```python
smoothed_value = α * previous_smoothed + (1 - α) * current_value
```

Where:
- `α = SMOOTHING_FACTOR` (default 0.7)
- Only applied if confidence > `MIN_CONFIDENCE_THRESHOLD`

**Purpose**: Reduces jitter and noise in gaze coordinates

### Blink Detection

**Algorithm**: Eye Aspect Ratio (EAR)

**Formula**:
```python
EAR = (vertical1 + vertical2) / (2.0 * horizontal)
```

Where:
- `vertical1`, `vertical2`: Vertical distances between eye landmarks
- `horizontal`: Horizontal distance between eye corners

**Thresholds**:
- EAR > 0.3: Eye open
- EAR < 0.2: Eye closed (blink)
- Used to adjust fusion weights during blinks

---

## Performance & Optimization

### Performance Bottlenecks

1. **MediaPipe Processing**: ~20-25ms per frame
2. **Head Pose Estimation**: ~2-3ms per frame
3. **Eye Gaze Calculation**: ~1-2ms per frame
4. **Visualization**: ~5-10ms per frame

**Total**: ~30-40ms per frame (25-33 FPS)

### Optimization Strategies

#### 1. Reduce Resolution
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```
**Impact**: 2-3x faster processing, slight accuracy loss

#### 2. Disable Iris Refinement
```python
REFINE_LANDMARKS = False
```
**Impact**: Faster MediaPipe processing, uses estimated iris position

#### 3. Increase Smoothing
```python
SMOOTHING_FACTOR = 0.9
```
**Impact**: Smoother output, but slower response to gaze changes

#### 4. Skip Frames
Process every Nth frame:
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    process_frame(frame)
```

#### 5. Reduce Calibration Points
Use 4-point calibration instead of 9:
```python
CALIBRATION_POINTS = [
    (0.1, 0.1), (0.9, 0.1),
    (0.1, 0.9), (0.9, 0.9)
]
```

### Memory Management

- Gaze data is stored in memory during operation
- Save data periodically to prevent memory buildup
- Calibration samples are cleared after model training
- MediaPipe automatically manages its internal buffers

### GPU Acceleration

MediaPipe can use GPU acceleration if available:
- Automatically detects and uses GPU
- No code changes required
- Can improve performance by 20-30%

---

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected

**Symptoms**: "Error: Could not open camera"

**Solutions**:
- Check camera is connected and not in use by another application
- Try different `CAMERA_ID` values (0, 1, 2, ...)
- Verify camera permissions on macOS/Linux
- Test camera with other applications

#### 2. Face Not Detected

**Symptoms**: No gaze point displayed, low confidence

**Solutions**:
- Ensure good lighting (face should be well-lit)
- Position face 50-70cm from camera
- Look directly at camera
- Remove obstructions (glasses, hats, masks)
- Lower `MIN_DETECTION_CONFIDENCE` to 0.5

#### 3. Poor Gaze Accuracy

**Symptoms**: Gaze point doesn't match where you're looking

**Solutions**:
- Run calibration (press `c`)
- Ensure stable head position during calibration
- Recalibrate if camera or position changes
- Check lighting conditions
- Verify camera focus is correct

#### 4. High CPU Usage

**Symptoms**: System becomes slow, low FPS

**Solutions**:
- Reduce camera resolution
- Disable iris refinement
- Increase smoothing factor
- Close other applications
- Process every other frame

#### 5. Calibration Fails

**Symptoms**: "Not enough calibration points" error

**Solutions**:
- Ensure face is detected during entire calibration
- Don't move head during calibration
- Ensure good lighting
- Try recalibrating
- Check that at least 4 points collected samples

#### 6. Jittery Gaze Point

**Symptoms**: Gaze point jumps around

**Solutions**:
- Increase `SMOOTHING_FACTOR` to 0.8-0.9
- Ensure stable lighting
- Check camera is stable (not shaking)
- Verify face detection is stable

#### 7. Low FPS

**Symptoms**: FPS below 20

**Solutions**:
- Reduce camera resolution
- Disable `REFINE_LANDMARKS`
- Close other applications
- Check CPU usage of other processes
- Use faster CPU or enable GPU acceleration

### Debug Mode

Enable verbose output by modifying the code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Profile the system to identify bottlenecks:

```python
import cProfile
cProfile.run('run_hybrid_gaze_tracking_with_calibration()')
```

---

## Future Enhancements

Based on the review document and system analysis, planned enhancements include:

### 1. Data Export Improvements
- [ ] Export JSON logger to CSV format
- [ ] Add .txt logger with metadata
- [ ] Include units in output
- [ ] Export calibration reports

### 2. Visualization Enhancements
- [ ] Real-time UI interface
- [ ] Render gaze coordinates with different colors
- [ ] Heatmap visualization
- [ ] Gaze path trails
- [ ] Fixation point indicators

### 3. Calibration Improvements
- [ ] Evaluate calibration process quality
- [ ] Identify uncertainties during calibration
- [ ] Report fault cases
- [ ] Automatic calibration quality assessment
- [ ] Detailed calibration documentation

### 4. Configuration Enhancements
- [ ] Sensitivity configuration levels (1-10)
- [ ] Per-user calibration profiles
- [ ] Adaptive parameter tuning
- [ ] Configuration presets

### 5. Advanced Features
- [ ] Deep learning-based gaze refinement
- [ ] Multi-user support
- [ ] Gaze-based UI control
- [ ] Attention analysis
- [ ] Saccade and fixation detection
- [ ] Real-time analytics dashboard

### 6. Integration
- [ ] Web-based interface (WebGazer.js integration)
- [ ] API for external applications
- [ ] ROS integration for robotics
- [ ] Unity/Unreal Engine plugins

### 7. Research Features
- [ ] A/B testing framework
- [ ] Statistical analysis tools
- [ ] Export to common eye-tracking formats (EDF, ASC)
- [ ] Integration with analysis software (EyeLink, Tobii)

---

## Additional Resources

### Documentation Files

- `README.md`: Quick start guide
- `architecture.md`: System architecture diagram
- `dfd.md`: Data flow diagram
- `sequence.md`: Sequence diagram of operations

### External Resources

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Citation

If using this system in research, please cite:

```
Gaze Tracker System - Hybrid Gaze Tracking with Calibration
Author: BalaJi
License: MIT
```

---

## License

Licensed under the **MIT License**.

You may use, modify, and distribute this project with attribution.

---

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Active Development

