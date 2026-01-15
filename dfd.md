# Data Flow Diagram (DFD)
```mermaid
flowchart TD

    %% Input
    Camera["📷 Camera<br>Input: Person in front of camera<br>Output: Live video frames"] -->|Captures frames| OpenCV["OpenCV Capture<br>Input: Video frames<br>Output: Face ROI frames"]

    OpenCV -->|Video frames| ScreenSetup["Screen Mapper Setup<br>Input: Video frames<br>Output: Screen corners coordinates"]
    ScreenSetup -->|Screen corners coordinates| FaceMesh["MediaPipe FaceMesh<br>Input: Frame with screen info<br>Output: Face & iris landmarks"]

    FaceMesh -->|Eye landmarks| EyeTracker["Eye Gaze Estimator<br>Input: Eye landmarks<br>Output: Raw gaze XY"]
    FaceMesh -->|Face landmarks| HeadPose["Head Pose Estimator<br>Input: Face landmarks<br>Output: Head direction vector"]

    EyeTracker -->|Raw gaze XY| GazeFusion["Gaze Fusion<br>Input: Eye gaze + Head direction<br>Output: Combined gaze vector"]
    HeadPose -->|Head direction vector| GazeFusion

    GazeFusion -->|Combined gaze| Calibration["Calibration Module<br>Input: Combined gaze vector<br>Output: Screen-mapped gaze XY"]
    Calibration -->|Calibrated gaze XY| Smoothing["Gaze Smoothing<br>Input: Calibrated gaze XY<br>Output: Smoothed gaze XY"]

    Smoothing -->|Stable gaze XY| ScreenMapper["Screen Coordinate Mapper<br>Input: Smoothed gaze XY + Screen corners<br>Output: Cursor position"]
    ScreenMapper -->|Cursor position| Cursor["Cursor or UI Output<br>Input: Cursor XY<br>Output: Visual cursor movement"]

    ScreenMapper -->|Gaze data| LogFile["gaze_logs<br>Input: Gaze & performance data<br>Output: CSV/JSON/TXT logs"]
    Smoothing -->|Performance stats| LogFile

    LogFile --> Developer["Developer or Analyst<br>Input: Logs<br>Output: Analysis & debugging"]
