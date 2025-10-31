---
title: Hybrid Gaze Tracking System Architecture
---
flowchart TD
    %% Input Layer
    subgraph InputLayer [📥 Input Layer]
        A[📷 Camera Hardware<br/>OpenCV VideoCapture]
        B[⌨️ User Input<br/>Keyboard Controls]
    end

    %% Processing Layer
    subgraph ProcessingLayer [⚙️ Processing Layer]
        C[👤 Face Detection<br/>MediaPipe Face Mesh]
        
        C --> D[🧠 Head Pose Estimation<br/>solvePnP + 3D Model]
        C --> E[👁️ Eye Gaze Estimation<br/>Iris Tracking]
        
        D --> F[🔄 Hybrid Fusion Engine]
        E --> F
        
        F --> G[📊 Smoothing & Filtering]
    end

    %% Calibration Subsystem
    subgraph CalibrationSubsystem [🎯 Calibration Subsystem]
        H[🎛️ Calibration Manager]
        
        H --> I[📍 Point-based Calibration<br/>9-point Grid]
        H --> J[🤖 Automatic Calibration<br/>Continuous Learning]
        
        I --> K[🧮 Calibration Model<br/>Polynomial Regression]
        J --> K
        
        K --> L[💾 Model Storage<br/>JSON Profiles]
    end

    %% Output Layer
    subgraph OutputLayer [📤 Output Layer]
        M[🖥️ Visualization & Display]
        N[📝 Data Logging & Storage]
        O[🎯 Gaze Coordinates Output]
    end

    %% Data Flow
    A --> C
    B --> H
    
    G --> O
    O --> M
    O --> N
    
    K -.-> F
    
    %% Styles
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef calibrationClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef outputClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B inputClass
    class C,D,E,F,G processClass
    class H,I,J,K,L calibrationClass
    class M,N,O outputClass
