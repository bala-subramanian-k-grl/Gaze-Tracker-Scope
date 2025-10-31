graph TB
    %% Input Layer
    subgraph InputLayer [Input Layer]
        A[Camera Hardware]
        B[User Input/Controls]
    end

    %% Processing Layer - Main Pipeline
    subgraph ProcessingLayer [Processing Layer]
        C[Face Detection & Landmark Extraction<br/>MediaPipe Face Mesh]
        
        subgraph ParallelProcessing [Parallel Processing]
            D[Head Pose Estimation<br/>solvePnP + 3D Model]
            E[Eye Gaze Estimation<br/>Iris Tracking + Eye Analysis]
        end
        
        F[Hybrid Fusion Engine<br/>Weighted Combination]
        G[Smoothing & Filtering<br/>Kalman-like Smoothing]
    end

    %% Calibration Subsystem
    subgraph CalibrationSubsystem [Calibration Subsystem]
        H[Calibration Manager]
        
        subgraph CalibrationModes [Calibration Modes]
            I[Point-based Calibration<br/>9-point Grid]
            J[Automatic Calibration<br/>Continuous Learning]
        end
        
        K[Calibration Model<br/>Polynomial Regression]
        L[Model Storage<br/>JSON Profiles]
    end

    %% Output Layer
    subgraph OutputLayer [Output Layer]
        M[Visualization & Display]
        N[Data Logging & Storage]
        O[Gaze Coordinates Output]
    end

    %% Data Flow Connections
    A --> C
    B --> H
    
    C --> D
    C --> E
    
    D --> F
    E --> F
    
    F --> G
    G --> O
    
    %% Calibration Connections
    H --> I
    H --> J
    I --> K
    J --> K
    K --> L
    K -.-> F
    
    %% Output Connections
    G --> M
    G --> N
    O --> M
    
    %% Style Definitions
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef calibrationClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef outputClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B inputClass
    class C,D,E,F,G processClass
    class H,I,J,K,L calibrationClass
    class M,N,O outputClass
