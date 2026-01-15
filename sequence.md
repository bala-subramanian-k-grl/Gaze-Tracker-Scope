# Sequence Diagram
```mermaid
sequenceDiagram
    participant User as User
    participant Camera
    participant CV as OpenCV
    participant SM as Screen Mapper Setup
    participant FM as MediaPipe FaceMesh
    participant ET as Eye Tracker
    participant HP as Head Pose Estimator
    participant GF as Gaze Fusion
    participant CAL as Calibration Module
    participant SMO as Gaze Smoothing
    participant MAP as Screen Mapper
    participant UI as Visualization UI
    participant Log as Log Files

    User ->> Camera: Looks at screen
    Camera ->> CV: Capture video frames

    CV ->> SM: Request screen corner setup
    User ->> SM: Click 4 screen corners
    SM -->> CV: Return screen boundaries

    CV ->> FM: Send frames for landmark detection
    FM ->> ET: Provide eye and iris landmarks
    FM ->> HP: Provide face landmarks

    ET -->> GF: Raw eye gaze
    HP -->> GF: Head direction

    GF ->> CAL: Combined gaze for calibration
    CAL -->> GF: Calibrated gaze model

    GF ->> SMO: Send calibrated gaze
    SMO -->> MAP: Stable gaze

    MAP ->> UI: Map gaze to screen and display
    MAP ->> Log: Save gaze data and stats

    UI -->> User: Show gaze cursor and status
