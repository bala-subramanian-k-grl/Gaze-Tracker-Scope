# 👁️ Gaze Tracking System

A hybrid **Gaze Tracking System** that captures and analyzes human eye gaze using **OpenCV** and **MediaPipe**.  
It supports both **manual point calibration** and **automatic calibration**, combining **head pose** and **eye gaze** data for accurate gaze estimation.

---

## 🚀 Features

- 🎥 **Real-Time Face & Eye Tracking** — Tracks facial landmarks and iris using MediaPipe.  
- 🧠 **Hybrid Fusion Engine** — Combines head pose and eye gaze vectors for robust gaze estimation.  
- ⚙️ **Calibration Modes**
  - **Point Calibration** (Manual)
  - **Automatic Calibration** (Adaptive)
- 📊 **Visualization** — Displays gaze direction and calibration feedback in real time.  
- 🪶 **Lightweight** — CPU-optimized, runs on standard webcams.  
- 📁 **Data Logging** — Optional CSV logging for gaze data analysis.  

---

## 🧩 System Architecture

```mermaid
flowchart TD
    A[Camera Input] --> B[MediaPipe Face Mesh]
    B --> C[Landmark Extraction]
    C --> D[Head Pose Estimation]
    C --> E[Eye Feature Extraction]
    C --> F[Iris Detection]
    D --> G[Head Gaze Vector]
    E --> H[Eye Gaze Vector]
    F --> H
    G --> I[Hybrid Fusion Engine]
    H --> I
    I --> J[Gaze Estimation Output]
    J --> K[Visualization and Logging]
```

---

## 🛠️ Installation

### Step 1 – Clone the Repository
```bash
git clone https://github.com/yourusername/gaze-tracking-system.git
cd gaze-tracking-system
```

### Step 2 – Create Virtual Environment (Optional)
```bash
python -m venv venv
# Activate it
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux
```

### Step 3 – Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Your `requirements.txt` should contain:
```
opencv-python
mediapipe
numpy
```

**Optional (for data logging and analysis):**
```
pandas
matplotlib
scikit-learn
scipy
```

---

## ▶️ Usage

### Run the Gaze Tracker
```bash
python gaze_tracker.py
```

### Keyboard Controls

| Key | Action |
|-----|---------|
| `c` | Start/Stop Calibration |
| `m` | Manual Point Calibration |
| `a` | Automatic Calibration |
| `q` | Quit Program |

---

## 🧠 Working Principle

1. **Face Detection** — MediaPipe detects 3D facial landmarks and iris positions.  
2. **Head Pose Estimation** — Calculates head orientation using selected landmarks.  
3. **Eye Gaze Vector** — Derived from iris center and eye corners.  
4. **Hybrid Fusion** — Merges head and eye gaze for stable tracking.  
5. **Calibration** — Maps gaze direction to screen coordinates.  
6. **Visualization** — Displays tracking overlay and calibration points.  

---

## 🗂️ Folder Structure

```
📁 gaze-tracking-system
│
├── gaze_tracker.py           # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── utils/
    ├── calibration.py        # Calibration module
    ├── fusion.py             # Hybrid fusion logic
    ├── visualization.py      # Real-time display
    └── logger.py             # Optional data logging
```

---

## 🌟 Future Enhancements

- 🔥 Heatmap visualization for gaze concentration  
- 📈 Real-time analytics dashboard  
- 🧩 Deep learning–based gaze refinement  
- 💻 Web-based gaze tracking (WebGazer.js integration)  
- 🎯 Eye-controlled UI navigation  

---

## 🧑‍💻 Author

**Bala Ji**  
💼 Victo Hosting | Victo Interns  
🌐 [victointern.site](https://victointern.site)

---

## 📄 License

Licensed under the **MIT License**.  
You may use, modify, and distribute this project with attribution.

---

## 🙌 Acknowledgements

- [MediaPipe by Google](https://developers.google.com/mediapipe)  
- [OpenCV](https://opencv.org/)  
- [WebGazer.js](https://webgazer.cs.brown.edu/) – concept inspiration  

---

⭐ **If you found this project useful, please give it a star on GitHub!**
