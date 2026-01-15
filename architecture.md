# System Architecture Diagram
```mermaid
graph TD

    %% Hardware
    subgraph Hardware
        Camera["📷 Camera
        Input: User face & eyes
        Output: Live video frames"]

        Computer["💻 Computer
        Input: Video frames
        Output: Sends to Software"]
    end

    %% Software
    subgraph Software
        OpenCV["🖼️ OpenCV
        Input: Video frames
        Task: Capture & preprocess frames
        Output: Frames for vision pipeline"]

        ScreenMapperSetup["🗺️ Screen Mapper (Mandatory Setup)
        Input: Mouse clicks (4 corners)
        Task: Define screen boundaries
        Output: Normalized screen map"]

        FaceMesh["🧠 MediaPipe FaceMesh
        Input: Frames
        Task: Detect face, eye & iris landmarks
        Output: Landmark coordinates"]

        EyeTracker["👁️ Eye Tracker
        Input: Eye & iris landmarks
        Task: Estimate raw eye gaze direction
        Output: Raw gaze (x, y)"]

        HeadPose["🧭 Head Pose Estimation
        Input: Face landmarks
        Task: Estimate head orientation
        Output: Head direction"]

        GazeFusion["🔀 Gaze Fusion
        Input: Eye gaze + Head direction
        Task: Combine both for stability
        Output: Combined gaze"]

        Calibration["🎯 Calibration System
        Input: Combined gaze + target points
        Task: Linear / Polynomial Regression
        Output: Calibrated gaze model"]

        Smoothing["🪶 Gaze Smoothing
        Input: Calibrated gaze
        Task: Reduce jitter & noise
        Output: Stable gaze"]

        ScreenMapper["📍 Screen Mapper
        Input: Stable gaze
        Task: Map gaze to screen pixels
        Output: Cursor position"]

        Visualizer["🖥️ Visualization
        Input: Frame + gaze data
        Task: Display gaze point & status"]
        
        Logger["📝 Logger
        Input: Gaze data & system stats
        Task: Log events, CPU & memory usage"]
    end

    %% Storage
    subgraph Storage
        LogFiles["📂 Log Files
        Stored: Gaze data,
        Calibration info,
        CPU & RAM usage"]
    end

    %% Connections
    Camera --> Computer
    Computer --> OpenCV

    OpenCV --> ScreenMapperSetup
    ScreenMapperSetup --> FaceMesh

    FaceMesh --> EyeTracker
    FaceMesh --> HeadPose
    EyeTracker --> GazeFusion
    HeadPose --> GazeFusion

    GazeFusion --> Calibration
    Calibration --> Smoothing
    Smoothing --> ScreenMapper
    ScreenMapper --> Visualizer

    ScreenMapper --> Logger
    Logger --> LogFiles
