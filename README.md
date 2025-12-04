# 👁️ Gaze Tracking System

A **Gaze Tracking System** that captures and analyzes human eye gaze using **OpenCV** and **MediaPipe**. 

---

## 🚀 Features

- **Real-time gaze tracking** with MediaPipe Face Mesh.
- **Head pose integration** for enhanced accuracy.
- **Eye gaze estimation** using iris and eye landmarks.
- **9-point calibration** (blink-based or OK button-based).
- **Screen boundary mapping** for precise screen coordinates.
- **Normal smoothing** for stable gaze points.
- **Blink detection** for calibration triggers.
- **System monitoring** (CPU, RAM usage) integrated in logs.
- **Logging** in JSON, CSV, and TXT formats.
- **Visualization** overlay with gaze points, crosshair, calibration targets, and warnings.
- Adjustable **cursor sensitivity** for smoothing responsiveness.

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
**Calibration Tips (Critical for Accuracy!)**

Sit ~50–70 cm from camera
Good, even lighting on your face (avoid backlight)
Look directly at each yellow circle
Keep head as still as possible during calibration
In blink mode: one clear blink per point when ready
In OK mode: look + click the green "OK" button







**Set Your Screen Boundaries (Highly Recommended!)**
Press s → Click these 4 corners in exact order:

Top-Left of your actual screen
Top-Right
Bottom-Right
Bottom-Left

This makes the gaze cursor land exactly where you're looking, even on laptops or external monitors!


**Calibrate (Critical for Accuracy!)**
Press c to start 9-point calibration.
You now have two options:
Option A – Blink Mode (Completely Hands-Free)

**Press b (default)**
Look at the yellow pulsing circle
Blink once clearly when ready
Wait for all 20 samples → automatically moves to next point
Repeat for all 9 points

Option B – OK Button Mode

**Press o**
Look at the circle
Click the green OK button when ready
Repeat for all 9 points

Calibration takes ~60–90 seconds.
Enjoy Smooth Gaze Control!

Move your eyes → cursor follows smoothly
Adjust speed with + and - keys (1 = slowest, 10 = fastest)
**Press l to toggle logging on/off**

**Press q anytime to quit and auto-save logs.**






### Keyboard Controls

| Key   | Function                               |
| ----- | ------------------------------------   |
| **q** | Quit                                   |
| **s** | Set screen boundaries                  |
| **c** | Start point calibration                |
| **r** | Reset calibration                      |
| **b** | Switch calibration to BLINK method     |
| **o** | Switch calibration to OK button method |
| **+** | Increase sensitivity                   |
| **-** | Decrease sensitivity                   |
| **l** | Toggle logging                         |


---

## 🧠 Working Principle

1. **Face Detection** — MediaPipe detects 3D facial landmarks and iris positions.  
2. **Head Pose Estimation** — Calculates head orientation using selected landmarks.  
3. **Eye Gaze Vector** — Derived from iris center and eye corners.  
4. **Hybrid Fusion** — Merges head and eye gaze for stable tracking.  
5. **Calibration** — Maps gaze direction to screen coordinates.  
6. **Visualization** — Displays tracking overlay and calibration points.  

---

## Common Issues & Warnings (Please Read Before Reporting Bugs)

| Issue / Warning                                   | Cause                                                      | Fix / Note                                                                 |
|---------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------|
| Cursor jumps all over the place                   | You skipped calibration or didn’t set screen corners (`s`) | Always press `s` → click 4 corners → press `c` → calibrate!                |
| Calibration never advances / stuck on point 1     | Using Blink mode but blinking too fast or too weakly       | Blink once clearly and wait 2–3 seconds per point                         |
| "OK" button does nothing                          | You are in Blink mode (`b`)                                Press `o` first to switch to OK-button mode                                  |
| Gaze is offset (e.g. center looks bottom-right)   | Screen corners not set or set in wrong order               Press `s` and click exactly: Top-Left → Top-Right → Bottom-Right → Bottom-Left |
| Very low confidence (0.00) all the time           | Wearing thick glasses, very low light, or extreme angle    | Improve lighting, remove heavy glasses, face camera straight                 |
| Program freezes or uses 100% CPU                   | Running on very old laptop or wrong camera resolution       | Try lowering camera resolution in code (e.g. 640×480)                       |
| Logs folder not created                            | No write permission in folder                               | Run terminal/Python as administrator or move script to Desktop/Documents    |

## 🗂️ Folder Structure

```
📁 gaze-tracking-system
│
├── gaze_tracker.py           # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── gaze_logs/                # auto-created logs (JSON, CSV, TXT)
    ├── gaze_data_20251120_185918.csv        
    ├── gaze_data_20251120_185918.json         
    ├── gaze_data_20251120_185918.txt  
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

**BalaJi**  

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
