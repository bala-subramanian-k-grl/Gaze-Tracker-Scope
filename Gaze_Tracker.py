
#!pip install opencv-python mediapipe numpy pandas matplotlib scipy

#Import libraries and configuration
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import logging
from typing import Tuple, Optional, Dict, Any, List
from scipy.spatial.transform import Rotation as R
import math
from enum import Enum
import csv
import os
import psutil
import threading
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Normal Smooth Configuration
class GazeConfig:
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 960
    FPS = 60

    # Cursor sensitivity (1 = slowest, 10 = fastest)
    CURSOR_SENSITIVITY = 1

    # MediaPipe settings for better accuracy
    MAX_NUM_FACES = 1
    REFINE_LANDMARKS = True
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7

    # Enhanced face landmarks for better head pose
    FACE_3D_POINTS = [1, 33, 263, 61, 291, 199]

    # Enhanced eye landmarks
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # Iris landmarks
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    # Normal smoothing - balanced between responsiveness and stability
    SMOOTHING_FACTOR = 0.5  # Balanced smoothing
    HEAD_SMOOTHING_FACTOR = 0.6

    # Optimized gaze estimation weights
    EYE_GAZE_WEIGHT = 0.85
    HEAD_GAZE_WEIGHT = 0.15
    MIN_CONFIDENCE_THRESHOLD = 0.4

    # Enhanced calibration
    CALIBRATION_POINTS = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
    ]
    CALIBRATION_DURATION = 2.5  # Slightly shorter
    CALIBRATION_SAMPLES_PER_POINT = 20

    # Logging settings
    ENABLE_LOGGING = True
    LOG_DIR = "gaze_logs"


config = GazeConfig()

# Create log directory
if config.ENABLE_LOGGING and not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)


# System Monitor
class SystemMonitor:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used_mb = 0.0

    def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        try:
            self.cpu_percent = psutil.cpu_percent(interval=0)
            memory_info = psutil.virtual_memory()
            self.memory_percent = memory_info.percent
            self.memory_used_mb = memory_info.used / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"System monitoring error: {e}")

        return {
            'cpu_percent': round(self.cpu_percent, 2),
            'memory_percent': round(self.memory_percent, 2),
            'memory_used_mb': round(self.memory_used_mb, 2)
        }


# Head Pose Estimator
class HeadPoseEstimator:
    def __init__(self, config: GazeConfig):
        self.config = config

        # 3D model points
        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0), # Right mouth corner
            (0.0, -330.0, -65.0)     # Chin
        ], dtype=np.float64)

        # Camera matrix
        focal_length = config.CAMERA_WIDTH
        center = (config.CAMERA_WIDTH / 2, config.CAMERA_HEIGHT / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))
        self.smoothed_rotation = np.zeros(3)
        self.smoothed_translation = np.zeros(3)

    def estimate_pose(self, landmarks):
        """Head pose estimation"""
        try:
            # Extract points
            image_points = []
            for idx in self.config.FACE_3D_POINTS:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    x = landmark.x * self.config.CAMERA_WIDTH
                    y = landmark.y * self.config.CAMERA_HEIGHT
                    image_points.append([x, y])

            if len(image_points) != len(self.model_3d_points):
                return np.zeros(3), np.zeros(3)

            image_points = np.array(image_points, dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_3d_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                euler_angles = self.rotation_matrix_to_euler_angles(rotation_mat)

                # Normal smoothing
                self.smoothed_rotation = (self.config.HEAD_SMOOTHING_FACTOR * self.smoothed_rotation +
                                          (1 - self.config.HEAD_SMOOTHING_FACTOR) * euler_angles)
                self.smoothed_translation = (self.config.HEAD_SMOOTHING_FACTOR * self.smoothed_translation +
                                             (1 - self.config.HEAD_SMOOTHING_FACTOR) * translation_vec.flatten())

                return self.smoothed_rotation, self.smoothed_translation

        except Exception:
            pass

        return np.zeros(3), np.zeros(3)

    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def head_pose_to_gaze_direction(self, rotation_angles):
        """Convert head pose to gaze direction"""
        pitch, yaw, roll = rotation_angles

        # Natural mapping
        head_gaze_x = 0.5 + (yaw / math.pi) * 0.8
        head_gaze_y = 0.5 + (pitch / math.pi) * 0.8

        head_gaze_x = np.clip(head_gaze_x, 0.0, 1.0)
        head_gaze_y = np.clip(head_gaze_y, 0.0, 1.0)

        return head_gaze_x, head_gaze_y


# Calibration System
class GazeCalibrator:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.calibration_model = None
        self.is_calibrated = False
        self.calibration_points = []
        self.raw_points = []

    def start_calibration(self):
        """Start calibration procedure"""
        self.calibration_points = []
        self.raw_points = []
        self.is_calibrated = False
        self.calibration_model = None
        print("Calibration started...")

    def add_calibration_sample(self, raw_gaze: Tuple[float, float], target_point: Tuple[float, float]):
        """Add calibration sample"""
        if (0 <= raw_gaze[0] <= 1 and 0 <= raw_gaze[1] <= 1 and
                0 <= target_point[0] <= 1 and 0 <= target_point[1] <= 1):
            self.raw_points.append(raw_gaze)
            self.calibration_points.append(target_point)

    def compute_calibration_model(self):
        """Compute calibration model"""
        if len(self.raw_points) < 6:
            print(f"Not enough calibration samples: {len(self.raw_points)}")
            return False

        X = np.array(self.raw_points)
        y = np.array(self.calibration_points)

        try:
            # Use linear regression for normal calibration
            model = LinearRegression()
            model.fit(X, y)

            self.calibration_model = model
            self.is_calibrated = True

            # Calculate calibration quality
            predictions = model.predict(X)
            errors = np.sqrt(np.sum((predictions - y) ** 2, axis=1))
            avg_error = np.mean(errors)

            print(f"Calibration success!")
            print(f"Average error: {avg_error:.4f}")
            print(f"Used {len(X)} samples")

            return True

        except Exception as e:
            print(f"Calibration error: {e}")
            return False

    def apply_calibration(self, raw_gaze_x: float, raw_gaze_y: float) -> Tuple[float, float]:
        """Apply calibration to raw gaze coordinates"""
        if not self.is_calibrated or self.calibration_model is None:
            return raw_gaze_x, raw_gaze_y

        try:
            input_coords = np.array([[raw_gaze_x, raw_gaze_y]])
            calibrated_coords = self.calibration_model.predict(input_coords)[0]

            calibrated_x = np.clip(calibrated_coords[0], 0.0, 1.0)
            calibrated_y = np.clip(calibrated_coords[1], 0.0, 1.0)

            return calibrated_x, calibrated_y

        except Exception:
            return raw_gaze_x, raw_gaze_y


# Data Logger with System Metrics
class DataLogger:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        """Setup JSON, CSV, and TXT logging with system metrics"""
        if not self.config.ENABLE_LOGGING:
            return

        # JSON file
        self.json_file = open(f"{self.config.LOG_DIR}/gaze_data_{self.session_id}.json", 'w')
        self.json_file.write('[\n')
        self.json_first_entry = True

        # CSV file
        self.csv_file = open(f"{self.config.LOG_DIR}/gaze_data_{self.session_id}.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'gaze_x', 'gaze_y', 'confidence', 'cpu_percent', 'memory_used_mb', 'memory_percent'])

        # TXT file (human readable)
        self.txt_file = open(f"{self.config.LOG_DIR}/gaze_data_{self.session_id}.txt", 'w')
        self.txt_file.write("Gaze Tracking Log with System Metrics\n")
        self.txt_file.write("=" * 50 + "\n")
        self.txt_file.write(f"Session: {self.session_id}\n")
        self.txt_file.write(f"Start Time: {datetime.now().isoformat()}\n")
        self.txt_file.write("=" * 50 + "\n\n")

        print(f"Logging started: {self.session_id}")

    def log_gaze_data(self, gaze_data: Dict):
        """Log gaze data in all formats with system metrics"""
        if not self.config.ENABLE_LOGGING:
            return

        try:
            # JSON logging
            if self.json_first_entry:
                self.json_first_entry = False
            else:
                self.json_file.write(',\n')
            json.dump(gaze_data, self.json_file, indent=2)

            # CSV logging
            self.csv_writer.writerow([
                gaze_data['timestamp'],
                gaze_data['gaze_x'],
                gaze_data['gaze_y'],
                gaze_data['confidence'],
                gaze_data['cpu_percent'],
                gaze_data['memory_used_mb'],
                gaze_data['memory_percent']
            ])

            # TXT logging
            self.txt_file.write(
                f"Time: {gaze_data['timestamp']} | "
                f"Gaze: ({gaze_data['gaze_x']}, {gaze_data['gaze_y']}) | "
                f"Conf: {gaze_data['confidence']:.3f} | "
                f"CPU: {gaze_data['cpu_percent']}% | "
                f"RAM: {gaze_data['memory_used_mb']}MB | "
                f"Mem%: {gaze_data['memory_percent']}%\n"
            )

        except Exception as e:
            print(f"Logging error: {e}")

    def close_logging(self):
        """Close all log files"""
        if not self.config.ENABLE_LOGGING:
            return

        try:
            # Close JSON file
            self.json_file.write('\n]')
            self.json_file.close()

            # Close CSV file
            self.csv_file.close()

            # Close TXT file
            self.txt_file.write(f"\nEnd Time: {datetime.now().isoformat()}\n")
            self.txt_file.close()

            print("Logging completed and files closed")

        except Exception as e:
            print(f"Error closing log files: {e}")


# Normal Smooth Gaze Tracker
class SmoothGazeTracker:
    def __init__(self, config: GazeConfig):
        self.config = config

        # Initialize components
        self.system_monitor = SystemMonitor(config)
        self.data_logger = DataLogger(config)

        # Initialize MediaPipe
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

        self.head_pose_estimator = HeadPoseEstimator(config)
        self.calibrator = GazeCalibrator(config)

        # State variables for normal smoothing
        self.smoothed_gaze_x = 0.5
        self.smoothed_gaze_y = 0.5
        self.smoothed_confidence = 0.0

        # Simple buffer for basic smoothing
        self.gaze_buffer_x = []
        self.gaze_buffer_y = []
        self.buffer_size = 3  # Small buffer for responsiveness

        # Calibration state
        self.calibrating = False
        self.current_calibration_point = 0
        self.calibration_start_time = 0

        # Calibration error handling
        self.GAZE_TOLERANCE = 0.15      # sensitivity for checking if user is looking correctly
        self.warning_active = False     # show warning on screen
        self.warning_timer = 0          # timeout for warning

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def is_gaze_correct(self, raw_x, raw_y, target_x, target_y):
        dx = abs(raw_x - target_x)
        dy = abs(raw_y - target_y)
        return dx <= self.GAZE_TOLERANCE and dy <= self.GAZE_TOLERANCE

    def get_eye_region(self, landmarks, eye_indices):
        """Extract eye region coordinates"""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * self.config.CAMERA_WIDTH)
                y = int(landmark.y * self.config.CAMERA_HEIGHT)
                eye_points.append((x, y))
        return eye_points

    def get_iris_center(self, landmarks, iris_indices):
        """Calculate iris center"""
        x_coords = []
        y_coords = []
        for idx in iris_indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x_coords.append(landmark.x * self.config.CAMERA_WIDTH)
                y_coords.append(landmark.y * self.config.CAMERA_HEIGHT)

        if x_coords:
            return (np.mean(x_coords), np.mean(y_coords))
        return None

    def apply_simple_smoothing(self, new_x, new_y, confidence):
        """Apply simple smoothing with small buffer"""
        self.gaze_buffer_x.append(new_x)
        self.gaze_buffer_y.append(new_y)

        # Keep buffer size limited
        if len(self.gaze_buffer_x) > self.buffer_size:
            self.gaze_buffer_x.pop(0)
            self.gaze_buffer_y.pop(0)

        # Simple average
        if self.gaze_buffer_x:
            avg_x = np.mean(self.gaze_buffer_x)
            avg_y = np.mean(self.gaze_buffer_y)
            return avg_x, avg_y

        return new_x, new_y

    def estimate_eye_gaze(self, eye_points, iris_center):
        """Estimate gaze direction from eye features"""
        # If iris center missing or too few eye points, return low confidence
        if iris_center is None or len(eye_points) < 6:
            return 0.5, 0.5, 0.0, 0.5, 0.5

        # Calculate eye bounding box
        eye_x = [p[0] for p in eye_points]
        eye_y = [p[1] for p in eye_points]

        eye_left = min(eye_x)
        eye_right = max(eye_x)
        eye_top = min(eye_y)
        eye_bottom = max(eye_y)

        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top

        if eye_width <= 0 or eye_height <= 0:
            return 0.5, 0.5, 0.0, 0.5, 0.5

        # Raw normalized coordinates (0..1) inside eye box
        raw_x = (iris_center[0] - eye_left) / eye_width
        raw_y = (iris_center[1] - eye_top) / eye_height

        # Natural mapping with sigmoid to make center stable
        gaze_x = 1 / (1 + np.exp(-8 * (raw_x - 0.5)))
        gaze_y = 1 / (1 + np.exp(-8 * (raw_y - 0.5)))

        gaze_x = np.clip(gaze_x, 0.0, 1.0)
        gaze_y = np.clip(gaze_y, 0.0, 1.0)

        # Confidence calculation - scaled by area but lower bounded
        area = eye_width * eye_height
        raw_conf = min(1.0, area / 600.0)
        # Give small positive confidence for reasonable small eyes to avoid permanent 0
        confidence = float(np.clip(raw_conf, 0.0, 1.0))

        return gaze_x, gaze_y, confidence, raw_x, raw_y

    def process_frame(self, frame):
        """Process frame with normal smooth tracking"""
        results = None
        try:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
        except Exception:
            # If processing fails, return current smoothed gaze and 0 confidence
            return frame, self.smoothed_gaze_x, self.smoothed_gaze_y, 0.0, {}

        # Default fallback values
        gaze_x, gaze_y, confidence = 0.5, 0.5, 0.0
        raw_gaze_x, raw_gaze_y = 0.5, 0.5
        debug_info = {}

        # --- Warning: no face detected ---
        if not results or not results.multi_face_landmarks:
            self.warning_active = True
            self.warning_timer = time.time()
            confidence = 0.0
            # Use smoothed gaze as fallback (so UI doesn't jump)
            return self.draw_and_return_frame_with_warning(frame, confidence)

        # If here, a face is detected — process landmarks
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Head pose estimation
            rotation_angles, _ = self.head_pose_estimator.estimate_pose(landmarks)
            head_gaze_x, head_gaze_y = self.head_pose_estimator.head_pose_to_gaze_direction(rotation_angles)

            # Eye tracking
            left_eye = self.get_eye_region(landmarks, self.config.LEFT_EYE_INDICES)
            right_eye = self.get_eye_region(landmarks, self.config.RIGHT_EYE_INDICES)

            left_iris = self.get_iris_center(landmarks, self.config.LEFT_IRIS_INDICES)
            right_iris = self.get_iris_center(landmarks, self.config.RIGHT_IRIS_INDICES)

            # Gaze estimation for both eyes
            left_x, left_y, left_conf, left_raw_x, left_raw_y = self.estimate_eye_gaze(left_eye, left_iris)
            right_x, right_y, right_conf, right_raw_x, right_raw_y = self.estimate_eye_gaze(right_eye, right_iris)

            # Average both eyes
            eye_gaze_x = (left_x + right_x) / 2
            eye_gaze_y = (left_y + right_y) / 2
            eye_confidence = (left_conf + right_conf) / 2
            raw_gaze_x = (left_raw_x + right_raw_x) / 2
            raw_gaze_y = (left_raw_y + right_raw_y) / 2

            # If irises or eyes missing -> mark confidence 0 and show warning
            if left_iris is None or right_iris is None or len(left_eye) < 6 or len(right_eye) < 6:
                self.warning_active = True
                self.warning_timer = time.time()
                confidence = 0.0
                # Use smoothed gaze as fallback
                gaze_x, gaze_y = self.smoothed_gaze_x, self.smoothed_gaze_y
            else:
                # Combine eye gaze and head pose
                combined_x = eye_gaze_x * self.config.EYE_GAZE_WEIGHT + head_gaze_x * self.config.HEAD_GAZE_WEIGHT
                combined_y = eye_gaze_y * self.config.EYE_GAZE_WEIGHT + head_gaze_y * self.config.HEAD_GAZE_WEIGHT

                # Apply calibration
                if self.calibrator.is_calibrated:
                    calibrated_x, calibrated_y = self.calibrator.apply_calibration(combined_x, combined_y)
                    gaze_x, gaze_y = calibrated_x, calibrated_y
                else:
                    gaze_x, gaze_y = combined_x, combined_y

                # Normalize confidence variable so it's available globally (was bug before)
                confidence = float(eye_confidence)

                # Normal smooth tracking with sensitivity if confidence sufficient
                if eye_confidence > self.config.MIN_CONFIDENCE_THRESHOLD:
                    # Apply simple buffer averaging
                    smoothed_x, smoothed_y = self.apply_simple_smoothing(gaze_x, gaze_y, eye_confidence)

                    # APPLY SENSITIVITY
                    sensitivity = np.clip(self.config.CURSOR_SENSITIVITY, 1, 10)
                    factor = sensitivity / 10.0   # convert 1–10 to 0.1–1

                    # Interpolate between previous & new gaze:
                    self.smoothed_gaze_x = self.smoothed_gaze_x * (1 - factor) + smoothed_x * factor
                    self.smoothed_gaze_y = self.smoothed_gaze_y * (1 - factor) + smoothed_y * factor

                    gaze_x, gaze_y = self.smoothed_gaze_x, self.smoothed_gaze_y
                else:
                    # If low confidence, don't update smoothed gaze to new noisy values
                    gaze_x, gaze_y = self.smoothed_gaze_x, self.smoothed_gaze_y

            # Handle calibration
            if self.calibrating:
                elapsed = time.time() - self.calibration_start_time

                if elapsed < self.config.CALIBRATION_DURATION:
                    target_point = self.config.CALIBRATION_POINTS[self.current_calibration_point]
                    self.calibrator.add_calibration_sample((raw_gaze_x, raw_gaze_y), target_point)
                else:
                    self.current_calibration_point += 1
                    self.calibration_start_time = time.time()

                    if self.current_calibration_point >= len(self.config.CALIBRATION_POINTS):
                        self.calibrating = False
                        success = self.calibrator.compute_calibration_model()
                        if success:
                            print("Calibration completed successfully!")
                        else:
                            print("Calibration failed. Please try again.")
                    else:
                        print(f"Moving to calibration point {self.current_calibration_point + 1}")

            # Get system stats
            system_stats = self.system_monitor.get_system_stats()

            # Prepare data for logging - convert normalized gaze to screen coords for logs
            screen_x = int(gaze_x * self.config.CAMERA_WIDTH)
            screen_y = int(gaze_y * self.config.CAMERA_HEIGHT)

            gaze_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "gaze_x": screen_x,
                "gaze_y": screen_y,
                "confidence": round(confidence, 3),
                "cpu_percent": system_stats['cpu_percent'],
                "memory_used_mb": system_stats['memory_used_mb'],
                "memory_percent": system_stats['memory_percent']
            }

            # Log the data
            self.data_logger.log_gaze_data(gaze_data)

            debug_info = {
                'eye_gaze': (float(eye_gaze_x), float(eye_gaze_y)),
                'head_gaze': (float(head_gaze_x), float(head_gaze_y)),
                'raw_gaze': (float(raw_gaze_x), float(raw_gaze_y)),
                'confidence': float(confidence),
                'calibrated': self.calibrator.is_calibrated,
                'screen_coords': (screen_x, screen_y),
                'system_stats': system_stats
            }

            # Visualization
            frame = self.draw_visualization(frame, left_eye, right_eye, left_iris, right_iris,
                                            screen_x, screen_y, confidence, system_stats)

        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0

        return frame, gaze_x, gaze_y, confidence, debug_info

    def draw_and_return_frame_with_warning(self, frame, confidence):
        """Helper to draw warning and return a frame when face/eyes missing"""
        # Use current smoothed gaze to avoid UI jump
        screen_x = int(self.smoothed_gaze_x * self.config.CAMERA_WIDTH)
        screen_y = int(self.smoothed_gaze_y * self.config.CAMERA_HEIGHT)

        #warning message for eye and face
        text = "WARNING: Face/Eyes not detected!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        h, w, _ = frame.shape
        x_center = int((w - text_size[0]) / 2)   # center horizontally
        y_top = 60                               # top position

        cv2.putText(
        frame,
        text,
        (x_center, y_top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


        # Auto hide warning after 2 sec
        if time.time() - self.warning_timer > 2:
            self.warning_active = False

        # Draw a faded gaze point for continuity
        cv2.circle(frame, (screen_x, screen_y), 8, (100, 100, 100), -1)
        cv2.circle(frame, (screen_x, screen_y), 12, (255, 255, 255), 2)

        # Show a small info block (FPS and conf)
        sys_stats = self.system_monitor.get_system_stats()
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Confidence: {confidence:.3f}",
            f"Calibrated: {self.calibrator.is_calibrated}"
        ]
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        return frame, self.smoothed_gaze_x, self.smoothed_gaze_y, confidence, {}

    def draw_visualization(self, frame, left_eye, right_eye, left_iris, right_iris, gaze_x, gaze_y, confidence, system_stats):
        """Draw visualization with system metrics"""
        h, w = frame.shape[:2]
        x_pos = int(w / 2 - 300) 
        y_pos = 50   



        if self.warning_active:
            cv2.putText(frame, "WARNING: Adjust your position / Look correctly!",
                        (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # auto hide warning after 2 sec
            if time.time() - self.warning_timer > 2:
                self.warning_active = False

        # Draw calibration points if calibrating
        if self.calibrating:
            for i, (px, py) in enumerate(self.config.CALIBRATION_POINTS):
                point_x = int(px * w)
                point_y = int(py * h)

                if i == self.current_calibration_point:
                    radius = 20 + int(10 * abs(math.sin(time.time() * 5)))
                    cv2.circle(frame, (point_x, point_y), radius, (0, 255, 255), 3)
                    cv2.putText(frame, "LOOK HERE", (point_x - 50, point_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.circle(frame, (point_x, point_y), 15, (100, 100, 100), 2)

        # Draw gaze point (gaze_x and gaze_y are passed as screen pixel coords)
        color = (0, 255, 0) if self.calibrator.is_calibrated else (255, 0, 255)

        try:
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 10, color, -1)
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 15, (255, 255, 255), 2)
        except Exception:
            pass

        # Simple crosshair
        crosshair_size = 25
        try:
            cv2.line(frame, (int(gaze_x - crosshair_size), int(gaze_y)), (int(gaze_x + crosshair_size), int(gaze_y)), (255, 255, 255), 2)
            cv2.line(frame, (int(gaze_x), int(gaze_y - crosshair_size)), (int(gaze_x), int(gaze_y + crosshair_size)), (255, 255, 255), 2)
        except Exception:
            pass

        # Draw eye contours
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        # Draw iris centers
        if left_iris:
            cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 0, 255), -1)
        if right_iris:
            cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 0, 255), -1)

        # Information display with system metrics
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Confidence: {confidence:.3f}",
            f"Calibrated: {self.calibrator.is_calibrated}",
            f"CPU: {system_stats['cpu_percent']}%",
            f"RAM: {system_stats['memory_used_mb']}MB ({system_stats['memory_percent']}%)"
        ]

        if self.calibrating:
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.config.CALIBRATION_DURATION - elapsed)
            info_lines.append(f"Cal Point: {self.current_calibration_point + 1}/{len(self.config.CALIBRATION_POINTS)}")
            info_lines.append(f"Time: {remaining:.1f}s")

        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        return frame

    def start_calibration(self):
        """Start calibration process"""
        self.calibrator.start_calibration()
        self.calibrating = True
        self.current_calibration_point = 0
        self.calibration_start_time = time.time()
        print("Calibration started!")
        print("Look directly at each yellow circle.")
        print(f"Calibration will take {len(self.config.CALIBRATION_POINTS) * self.config.CALIBRATION_DURATION:.1f} seconds")

    def close(self):
        """Cleanup resources"""
        self.data_logger.close_logging()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Main function
def run_smooth_gaze_tracking():
    """Run smooth gaze tracking with comprehensive logging"""

    config = GazeConfig()

    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")

    # Update config with actual values
    config.CAMERA_WIDTH = actual_width
    config.CAMERA_HEIGHT = actual_height

    # Initialize smooth tracker
    gaze_tracker = SmoothGazeTracker(config)

    print("\n" + "=" * 60)
    print("GAZE TRACKING WITH SYSTEM METRICS & LOGGING")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit and save logs")
    print("  'c' - Start calibration")
    print("  'r' - Reset calibration")
    print("  'l' - Toggle logging")
    print("\nFor best accuracy:")
    print("1. Good lighting on your face")
    print("2. Face camera directly")
    print("3. Calibrate first (press 'c')")
    print("4. Keep head still during calibration")
    print(f"5. Logs saved to: {config.LOG_DIR}/")
    print("=" * 60)

    logging_enabled = config.ENABLE_LOGGING

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            processed_frame, gaze_x, gaze_y, confidence, debug_info = gaze_tracker.process_frame(frame)

            # Display
            cv2.imshow('Smooth Gaze Tracking - Press Q to quit', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                gaze_tracker.start_calibration()
            elif key == ord('r'):
                gaze_tracker.calibrator.is_calibrated = False
                gaze_tracker.calibrator.calibration_model = None
                gaze_tracker.smoothed_gaze_x = 0.5
                gaze_tracker.smoothed_gaze_y = 0.5
                gaze_tracker.gaze_buffer_x = []
                gaze_tracker.gaze_buffer_y = []
                print("Calibration reset")
            elif key == ord('l'):
                logging_enabled = not logging_enabled
                gaze_tracker.config.ENABLE_LOGGING = logging_enabled
                print(f"Logging: {'ENABLED' if logging_enabled else 'DISABLED'}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        gaze_tracker.close()

        print("\n" + "=" * 60)
        print("SESSION COMPLETED - SMOOTH TRACKING")
        print("=" * 60)
        print(f"Total frames processed: {gaze_tracker.frame_count}")
        print(f"Average FPS: {gaze_tracker.fps:.1f}")
        print(f"Final calibration status: {gaze_tracker.calibrator.is_calibrated}")
        if config.ENABLE_LOGGING:
            print(f"Log files saved to: {config.LOG_DIR}/")
            print("File formats: JSON, CSV, TXT with system metrics")
        print("=" * 60)


# Run the smooth gaze tracker
if __name__ == "__main__":
    run_smooth_gaze_tracking()
