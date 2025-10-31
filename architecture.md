graph TB
    %% Main System Components
    SUB[Hybrid Gaze Tracking System]
    
    %% Input Layer
    CAM[Camera Input<br/>RGB Frame]
    
    %% Core Processing Pipeline
    MP[MediaPipe Face Mesh<br/>Facial Landmarks Detection]
    HP[Head Pose Estimator<br/>solvePnP + Euler Angles]
    EG[Eye Gaze Estimator<br/>Iris Position + Eye Geometry]
    
    %% Data Processing Components
    CALIB[Gaze Calibrator<br/>Point/Automatic Calibration]
    COMB[Gaze Combiner<br/>Weighted Fusion]
    SMOOTH[Smoothing Filter<br/>Exponential Smoothing]
    
    %% Calibration Subsystem
    subgraph CalibrationSystem
        PCAL[Point Calibration<br/>9-Point Grid]
        ACAL[Automatic Calibration<br/>Continuous Learning]
        CMOD[Calibration Model<br/>Polynomial Regression]
    end
    
    %% Output Layer
    VIS[Visualization Engine<br/>Real-time Display]
    DATA[Data Logger<br/>JSON Export]
    OUT[Gaze Coordinates<br/>x, y, confidence]
    
    %% Configuration
    CONF[GazeConfig<br/>System Parameters]
    
    %% Data Flow
    CAM --> MP
    MP --> HP
    MP --> EG
    
    HP --> COMB
    EG --> COMB
    
    COMB --> CALIB
    CALIB --> SMOOTH
    SMOOTH --> OUT
    
    CALIB --> PCAL
    CALIB --> ACAL
    PCAL --> CMOD
    ACAL --> CMOD
    CMOD --> CALIB
    
    OUT --> VIS
    OUT --> DATA
    
    CONF -.-> MP
    CONF -.-> HP
    CONF -.-> EG
    CONF -.-> CALIB
    CONF -.-> SMOOTH
    
    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef config fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef calibration fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class CAM input
    class MP,HP,EG,COMB,SMOOTH process
    class VIS,OUT,DATA output
    class CONF config
    class PCAL,ACAL,CMOD,CALIB calibration
