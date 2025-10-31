# 👁️ Gaze Tracking System

A hybrid **Gaze Tracking System** that captures and analyzes human eye gaze using **OpenCV** and **MediaPipe**.  
This system supports both **manual point calibration** and **automatic calibration**, combining **head pose** and **eye gaze** data for accurate gaze estimation.

---

## 🚀 Features

- 🎥 **Real-Time Face & Eye Tracking** — Tracks face landmarks and iris in real-time using MediaPipe.
- 🧠 **Hybrid Fusion Engine** — Combines head pose and eye gaze vectors for robust gaze estimation.
- ⚙️ **Calibration Modes**
  - **Point Calibration** (Manual)
  - **Automatic Calibration** (Adaptive)
- 📊 **Visualization** — Displays gaze direction, calibration points, and tracking lines.
- 🪶 **Lightweight** — CPU-optimized, runs on standard webcams.
- 📁 **Data Logging** — Stores gaze data for research and analysis.

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
    J --> K[Visualization & Logging]
🛠️ Installation
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/gaze-tracking-system.git
cd gaze-tracking-system
2️⃣ Create Virtual Environment (Optional)
bash
Copy code
python -m venv venv
# Activate it
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
📦 Requirements
Include these in your requirements.txt file:

Copy code
opencv-python
mediapipe
numpy
(Optional for advanced features:)

nginx
Copy code
pandas
matplotlib
scikit-learn
scipy
▶️ Usage
Run the Gaze Tracker
bash
Copy code
python gaze_tracker.py
Controls
Key	Action
c	Start/Stop Calibration
m	Switch to Manual Point Calibration
a	Switch to Automatic Calibration
q	Quit Program

🧠 Working Principle
Face & Eye Detection — MediaPipe detects 3D facial landmarks and iris positions.

Head Pose Estimation — Uses key facial landmarks to compute head orientation.

Eye Gaze Vector — Derived from the iris center and eye landmarks.

Hybrid Fusion — Combines eye and head vectors for stable gaze direction.

Calibration — Maps gaze direction to screen coordinates.

Visualization — Displays tracking overlay, calibration dots, and gaze estimation lines.

🗂️ Folder Structure
perl
Copy code
📁 gaze-tracking-system
│
├── gaze_tracker.py           # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── utils/
    ├── calibration.py        # Calibration module
    ├── fusion.py             # Hybrid fusion logic
    ├── visualization.py      # Real-time drawing functions
    └── logger.py             # CSV logging utilities
🌟 Future Enhancements
🔥 Heatmap visualization for gaze concentration

📈 Real-time analytics dashboard

🧩 Deep learning–based gaze refinement

💻 Web-based gaze tracking (WebGazer.js integration)

🎯 Gaze-controlled interface for accessibility

🧑‍💻 Author
Bala Ji
💼 Victo Hosting | Victo Interns
🌐 victointern.site

📄 License
This project is licensed under the MIT License.
You’re free to use, modify, and distribute it with attribution.

🙌 Acknowledgements
MediaPipe by Google

OpenCV

WebGazer.js (concept inspiration)

