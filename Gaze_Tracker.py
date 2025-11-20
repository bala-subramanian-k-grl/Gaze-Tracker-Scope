!pip install opencv-python mediapipe numpy pandas matplotlib scipy psutil

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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from enum import Enum
import csv
import os
import psutil
import threading

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Calibration modes
class CalibrationMode(Enum):
    NONE = 0
    POINT_CALIBRATION = 1
    AUTOMATIC_CALIBRATION = 2

# Enhanced Configuration with logging
class GazeConfig:
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    FPS = 30
    
    # MediaPipe settings
    MAX_NUM_FACES = 1
    REFINE_LANDMARKS = True
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Face landmarks for head pose estimation
    FACE_3D_POINTS = [1, 33, 263, 61, 291, 199]
    
    # Eye landmarks
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Iris landmarks
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    # Smoothing
    SMOOTHING_FACTOR = 0.7
    HEAD_SMOOTHING_FACTOR = 0.8
    
    # Gaze estimation parameters
    EYE_GAZE_WEIGHT = 0.7
    HEAD_GAZE_WEIGHT = 0.3
    MIN_CONFIDENCE_THRESHOLD = 0.4
    
    # Calibration settings
    CALIBRATION_POINTS = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row  
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
    ]
    CALIBRATION_DURATION = 3.0  # seconds per point
    CALIBRATION_SAMPLES_PER_POINT = 30
    
    # System monitoring settings
    SYSTEM_MONITOR_INTERVAL = 2.0  # seconds between system monitoring updates
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ENABLE_FILE_LOGGING = True

config = GazeConfig()

# System Monitor Class
class SystemMonitor:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.cpu_percent = 0.0
        self.memory_used_mb = 0.0
        self.memory_percent = 0.0
        self.is_monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger('SystemMonitor')
        
    def start_monitoring(self):
        """Start system monitoring in a separate thread"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("System monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get CPU usage (blocking call, first call might return 0)
                self.cpu_percent = psutil.cpu_percent(interval=0.5)
                
                # Get memory usage
                memory_info = psutil.virtual_memory()
                self.memory_used_mb = memory_info.used / (1024 * 1024)  # Convert to MB
                self.memory_percent = memory_info.percent
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                
            time.sleep(self.config.SYSTEM_MONITOR_INTERVAL)
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_percent': self.memory_percent
        }

# Enhanced Head Pose Estimator Class
class HeadPoseEstimator:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.logger = logging.getLogger('HeadPoseEstimator')
        
        # 3D model points for head pose estimation (in mm)
        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0), # Right mouth corner
            (0.0, -330.0, -65.0)     # Chin
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = config.CAMERA_WIDTH
        center = (config.CAMERA_WIDTH/2, config.CAMERA_HEIGHT/2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (assuming no distortion)
        self.dist_coeffs = np.zeros((4, 1))
        
        # Smoothing
        self.smoothed_rotation = np.zeros(3)
        self.smoothed_translation = np.zeros(3)
        
    def estimate_pose(self, landmarks):
        """Estimate head pose using solvePnP"""
        try:
            # Extract 2D image points
            image_points = []
            for idx in self.config.FACE_3D_POINTS:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    x = landmark.x * self.config.CAMERA_WIDTH
                    y = landmark.y * self.config.CAMERA_HEIGHT
                    image_points.append([x, y])
            
            if len(image_points) != len(self.model_3d_points):
                return np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)
                
            image_points = np.array(image_points, dtype=np.float64)
            
            # Solve PnP problem
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_3d_points, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                
                # Convert to Euler angles
                euler_angles = self.rotation_matrix_to_euler_angles(rotation_mat)
                
                # Apply smoothing
                self.smoothed_rotation = (self.config.HEAD_SMOOTHING_FACTOR * self.smoothed_rotation + 
                                        (1 - self.config.HEAD_SMOOTHING_FACTOR) * euler_angles)
                self.smoothed_translation = (self.config.HEAD_SMOOTHING_FACTOR * self.smoothed_translation + 
                                           (1 - self.config.HEAD_SMOOTHING_FACTOR) * translation_vec.flatten())
                
                return self.smoothed_rotation, self.smoothed_translation, rotation_mat, translation_vec
                
        except Exception as e:
            self.logger.error(f"Head pose estimation error: {e}")
        
        return np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def head_pose_to_gaze_direction(self, rotation_angles, translation_vec):
        """Convert head pose to approximate gaze direction"""
        # Extract angles (in radians)
        pitch, yaw, roll = rotation_angles
        
        # Convert to gaze coordinates (normalized 0-1)
        head_gaze_x = 0.5 + (yaw / math.pi) * 0.8
        head_gaze_y = 0.5 + (pitch / math.pi) * 0.8
        
        # Clamp values
        head_gaze_x = np.clip(head_gaze_x, 0.0, 1.0)
        head_gaze_y = np.clip(head_gaze_y, 0.0, 1.0)
        
        return head_gaze_x, head_gaze_y

# Enhanced Calibration System with logging
class GazeCalibrator:
    def __init__(self, config: GazeConfig):
        self.config = config
        self.calibration_data = {}
        self.calibration_model = None
        self.is_calibrated = False
        self.logger = logging.getLogger('GazeCalibrator')
        
    def start_point_calibration(self):
        """Start point-based calibration procedure"""
        self.calibration_mode = CalibrationMode.POINT_CALIBRATION
        self.current_calibration_point = 0
        self.calibration_start_time = time.time()
        self.calibration_samples = {i: [] for i in range(len(self.config.CALIBRATION_POINTS))}
        self.logger.info("Starting point calibration...")
        
    def start_automatic_calibration(self):
        """Start automatic calibration (collects data during normal use)"""
        self.calibration_mode = CalibrationMode.AUTOMATIC_CALIBRATION
        self.calibration_samples = {'auto': []}
        self.logger.info("Starting automatic calibration...")
        
    def add_calibration_sample(self, point_index: int, raw_gaze_data: Dict, screen_point: Tuple[float, float]):
        """Add a calibration sample for the current point"""
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION:
            if point_index in self.calibration_samples:
                sample = {
                    'timestamp': time.time(),
                    'raw_gaze': raw_gaze_data,
                    'target_point': screen_point,
                    'raw_gaze_x': raw_gaze_data.get('raw_gaze_x', 0),
                    'raw_gaze_y': raw_gaze_data.get('raw_gaze_y', 0),
                    'eye_features': raw_gaze_data.get('eye_features', {}),
                    'head_pose': raw_gaze_data.get('head_pose', {})
                }
                self.calibration_samples[point_index].append(sample)
                
        elif self.calibration_mode == CalibrationMode.AUTOMATIC_CALIBRATION:
            sample = {
                'timestamp': time.time(),
                'raw_gaze': raw_gaze_data,
                'target_point': screen_point,
                'raw_gaze_x': raw_gaze_data.get('raw_gaze_x', 0),
                'raw_gaze_y': raw_gaze_data.get('raw_gaze_y', 0)
            }
            self.calibration_samples['auto'].append(sample)
    
    def compute_calibration_model(self):
        """Compute calibration model from collected samples"""
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION:
            return self._compute_point_calibration_model()
        elif self.calibration_mode == CalibrationMode.AUTOMATIC_CALIBRATION:
            return self._compute_automatic_calibration_model()
        return None
    
    def _compute_point_calibration_model(self):
        """Compute calibration model from point calibration data"""
        all_inputs = []
        all_targets = []
        
        for point_idx, samples in self.calibration_samples.items():
            if len(samples) < 5:  # Need minimum samples per point
                continue
                
            # Average samples for each calibration point
            avg_gaze_x = np.mean([s['raw_gaze_x'] for s in samples])
            avg_gaze_y = np.mean([s['raw_gaze_y'] for s in samples])
            target_point = samples[0]['target_point']
            
            all_inputs.append([avg_gaze_x, avg_gaze_y])
            all_targets.append([target_point[0], target_point[1]])
        
        if len(all_inputs) < 4:  # Need at least 4 points for good calibration
            self.logger.warning(f"Not enough calibration points: {len(all_inputs)}")
            return None
        
        # Convert to numpy arrays
        X = np.array(all_inputs)
        y = np.array(all_targets)
        
        try:
            # Use polynomial regression for better mapping
            # Add polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            
            # Create polynomial regression model
            degree = 2
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            
            # Fit the model
            model.fit(X, y)
            
            # Test the model
            predictions = model.predict(X)
            mse = np.mean((predictions - y) ** 2)
            self.logger.info(f"Calibration model trained. MSE: {mse:.4f}")
            
            self.is_calibrated = True
            self.calibration_model = model
            return model
            
        except Exception as e:
            self.logger.error(f"Error computing calibration model: {e}")
            return None
    
    def _compute_automatic_calibration_model(self):
        """Compute calibration model from automatic calibration data"""
        # This would use more advanced techniques like clustering
        # For now, use simple linear regression
        if 'auto' not in self.calibration_samples or len(self.calibration_samples['auto']) < 20:
            self.logger.warning("Not enough automatic calibration samples")
            return None
            
        samples = self.calibration_samples['auto']
        X = np.array([[s['raw_gaze_x'], s['raw_gaze_y']] for s in samples])
        y = np.array([s['target_point'] for s in samples])
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        self.is_calibrated = True
        self.calibration_model = model
        return model
    
    def apply_calibration(self, raw_gaze_x: float, raw_gaze_y: float) -> Tuple[float, float]:
        """Apply calibration model to raw gaze coordinates"""
        if not self.is_calibrated or self.calibration_model is None:
            return raw_gaze_x, raw_gaze_y
        
        try:
            input_coords = np.array([[raw_gaze_x, raw_gaze_y]])
            calibrated_coords = self.calibration_model.predict(input_coords)[0]
            calibrated_x = np.clip(calibrated_coords[0], 0.0, 1.0)
            calibrated_y = np.clip(calibrated_coords[1], 0.0, 1.0)
            return calibrated_x, calibrated_y
        except Exception as e:
            self.logger.error(f"Calibration application error: {e}")
            return raw_gaze_x, raw_gaze_y
    
    def save_calibration(self, filename: str = None):
        """Save calibration data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_calibration_{timestamp}.json"
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'is_calibrated': self.is_calibrated,
            'calibration_mode': self.calibration_mode.name if hasattr(self, 'calibration_mode') else 'NONE',
            'model_coefficients': self._get_model_coefficients(),
            'calibration_samples_summary': {
                str(k): len(v) for k, v in self.calibration_samples.items()
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            self.logger.info(f"Calibration saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving calibration: {e}")
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            # For simplicity, we're just loading the status
            # In a full implementation, you'd reconstruct the model
            self.is_calibrated = calibration_data.get('is_calibrated', False)
            self.logger.info(f"Calibration loaded: {self.is_calibrated}")
            return self.is_calibrated
            
        except Exception as e:
            self.logger.error(f"Error loading calibration: {e}")
            return False
    
    def _get_model_coefficients(self):
        """Extract model coefficients for saving"""
        if self.calibration_model is None:
            return None
        
        try:
            if hasattr(self.calibration_model, 'named_steps'):
                # Pipeline model
                linear_model = self.calibration_model.named_steps['linear']
                poly = self.calibration_model.named_steps['poly']
                return {
                    'coefficients': linear_model.coef_.tolist(),
                    'intercept': linear_model.intercept_.tolist(),
                    'poly_features': poly.powers_.tolist()
                }
            else:
                # Simple linear model
                return {
                    'coefficients': self.calibration_model.coef_.tolist(),
                    'intercept': self.calibration_model.intercept_.tolist()
                }
        except:
            return None

# Enhanced Hybrid Gaze Tracker with Calibration and Logging
class HybridGazeTracker:
    def __init__(self, config: GazeConfig):
        self.config = config
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger('HybridGazeTracker')
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor(config)
        self.system_monitor.start_monitoring()
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.head_pose_estimator = HeadPoseEstimator(config)
        self.calibrator = GazeCalibrator(config)
        
        # State variables
        self.smoothed_combined_gaze_x = 0.5
        self.smoothed_combined_gaze_y = 0.5
        
        # Calibration state
        self.calibration_mode = CalibrationMode.NONE
        self.current_calibration_point = 0
        self.calibration_start_time = 0
        self.calibration_collection_active = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Data storage
        self.gaze_data = []
        self.csv_writer = None
        self.csv_file = None
        
        self.logger.info("Hybrid Gaze Tracker with Calibration and System Monitoring initialized successfully!")
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"gaze_tracking_{timestamp}.log"
        
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format=self.config.LOG_FORMAT,
            handlers=[]
        )
        
        # Create logger
        logger = logging.getLogger()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.LOG_LEVEL)
        console_handler.setFormatter(logging.Formatter(self.config.LOG_FORMAT))
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.ENABLE_FILE_LOGGING:
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setLevel(self.config.LOG_LEVEL)
            file_handler.setFormatter(logging.Formatter(self.config.LOG_FORMAT))
            logger.addHandler(file_handler)
    
    def setup_csv_logging(self, filename=None):
        """Setup CSV logging for gaze data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_data_{timestamp}.csv"
        
        try:
            self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header with system monitoring columns
            header = [
                'timestamp', 'gaze_x', 'gaze_y', 'confidence', 'fps',
                'eye_gaze_x', 'eye_gaze_y', 'head_gaze_x', 'head_gaze_y',
                'raw_gaze_x', 'raw_gaze_y', 'eye_confidence', 'eye_closure_ratio',
                'eye_weight', 'head_weight', 'pitch_deg', 'yaw_deg', 'roll_deg',
                'calibrated', 'resolution', 'cpu_percent', 'memory_used_mb', 'memory_percent'
            ]
            self.csv_writer.writerow(header)
            self.logger.info(f"CSV logging started: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error setting up CSV logging: {e}")
    
    def write_csv_row(self, gaze_data):
        """Write a row to CSV file"""
        if self.csv_writer is None:
            return
        
        try:
            row = [
                gaze_data['timestamp'],
                gaze_data['gaze_x'],
                gaze_data['gaze_y'],
                gaze_data['confidence'],
                gaze_data['fps'],
                gaze_data['eye_gaze_x'],
                gaze_data['eye_gaze_y'],
                gaze_data['head_gaze_x'],
                gaze_data['head_gaze_y'],
                gaze_data['raw_gaze_x'],
                gaze_data['raw_gaze_y'],
                gaze_data['eye_confidence'],
                gaze_data['eye_closure_ratio'],
                gaze_data['eye_weight'],
                gaze_data['head_weight'],
                np.degrees(gaze_data['head_pose_angles'][0]) if gaze_data['head_pose_angles'] else 0,
                np.degrees(gaze_data['head_pose_angles'][1]) if gaze_data['head_pose_angles'] else 0,
                np.degrees(gaze_data['head_pose_angles'][2]) if gaze_data['head_pose_angles'] else 0,
                gaze_data['calibrated'],
                gaze_data['resolution'],
                gaze_data['cpu_percent'],
                gaze_data['memory_used_mb'],
                gaze_data['memory_percent']
            ]
            self.csv_writer.writerow(row)
            
        except Exception as e:
            self.logger.error(f"Error writing CSV row: {e}")
    
    def close_csv_logging(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            self.logger.info("CSV logging closed")
    
    def start_calibration(self, mode: str = "point"):
        """Start calibration procedure"""
        if mode == "point":
            self.calibrator.start_point_calibration()
            self.calibration_mode = CalibrationMode.POINT_CALIBRATION
            self.current_calibration_point = 0
            self.calibration_start_time = time.time()
            self.calibration_collection_active = True
            self.logger.info("Point calibration started")
        elif mode == "auto":
            self.calibrator.start_automatic_calibration()
            self.calibration_mode = CalibrationMode.AUTOMATIC_CALIBRATION
            self.calibration_collection_active = True
            self.logger.info("Automatic calibration started")
    
    def update_calibration(self):
        """Update calibration state machine"""
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION:
            elapsed = time.time() - self.calibration_start_time
            
            if elapsed > self.config.CALIBRATION_DURATION:
                # Move to next point
                self.current_calibration_point += 1
                self.calibration_start_time = time.time()
                
                if self.current_calibration_point >= len(self.config.CALIBRATION_POINTS):
                    # Calibration complete
                    self.complete_calibration()
    
    def complete_calibration(self):
        """Complete calibration and compute model"""
        self.calibration_collection_active = False
        self.calibration_mode = CalibrationMode.NONE
        
        self.logger.info("Calibration completed. Computing model...")
        model = self.calibrator.compute_calibration_model()
        
        if model is not None:
            self.logger.info("Calibration model computed successfully!")
            self.calibrator.save_calibration()
        else:
            self.logger.warning("Failed to compute calibration model")
    
    def get_current_calibration_point(self) -> Tuple[float, float]:
        """Get current calibration point coordinates"""
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION:
            if self.current_calibration_point < len(self.config.CALIBRATION_POINTS):
                return self.config.CALIBRATION_POINTS[self.current_calibration_point]
        return (0.5, 0.5)
    
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
        """Calculate iris center with high precision"""
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = landmark.x * self.config.CAMERA_WIDTH
                y = landmark.y * self.config.CAMERA_HEIGHT
                iris_points.append((x, y))
        
        if iris_points:
            center_x = np.mean([p[0] for p in iris_points])
            center_y = np.mean([p[1] for p in iris_points])
            return (center_x, center_y)
        return None
    
    def calculate_eye_closure_ratio(self, eye_points):
        """Calculate eye closure ratio for blink detection"""
        if len(eye_points) < 6:
            return 1.0
            
        # Vertical distances
        vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Horizontal distance
        horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if horizontal == 0:
            return 1.0
            
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def estimate_eye_gaze_direction(self, eye_points, iris_center):
        """Estimate gaze direction based on iris position within eye region"""
        if iris_center is None or len(eye_points) < 6:
            return 0.5, 0.5, 0.0
        
        # Calculate eye bounding box
        eye_x = [p[0] for p in eye_points]
        eye_y = [p[1] for p in eye_points]
        
        eye_left = min(eye_x)
        eye_right = max(eye_x)
        eye_top = min(eye_y)
        eye_bottom = max(eye_y)
        
        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top
        
        if eye_width == 0 or eye_height == 0:
            return 0.5, 0.5, 0.0
        
        # Normalize iris position within eye region
        raw_gaze_x = (iris_center[0] - eye_left) / eye_width
        raw_gaze_y = (iris_center[1] - eye_top) / eye_height
        
        # Apply non-linear mapping for more natural movement
        gaze_x = 1 / (1 + np.exp(-8 * (raw_gaze_x - 0.5)))
        gaze_y = 1 / (1 + np.exp(-8 * (raw_gaze_y - 0.5)))
        
        # Clamp values
        gaze_x = np.clip(gaze_x, 0.0, 1.0)
        gaze_y = np.clip(gaze_y, 0.0, 1.0)
        
        # Calculate confidence based on eye detection quality
        confidence = min(1.0, (eye_width * eye_height) / 800.0)
        
        return gaze_x, gaze_y, confidence, raw_gaze_x, raw_gaze_y
    
    def combine_gaze_estimates(self, eye_gaze_x, eye_gaze_y, eye_confidence, 
                             head_gaze_x, head_gaze_y, eye_closure_ratio):
        """Combine eye gaze and head pose estimates"""
        
        # Adjust weights based on eye closure (blink detection)
        if eye_closure_ratio < 0.2:  # Eye is closed
            eye_weight = 0.1
            head_weight = 0.9
        else:
            # Use confidence-based weighting
            eye_weight = self.config.EYE_GAZE_WEIGHT * min(eye_confidence, 1.0)
            head_weight = self.config.HEAD_GAZE_WEIGHT
        
        # Normalize weights
        total_weight = eye_weight + head_weight
        if total_weight > 0:
            eye_weight /= total_weight
            head_weight /= total_weight
        
        # Combine estimates
        combined_x = eye_gaze_x * eye_weight + head_gaze_x * head_weight
        combined_y = eye_gaze_y * eye_weight + head_gaze_y * head_weight
        
        # Overall confidence
        combined_confidence = (eye_confidence * eye_weight + head_weight)
        
        return combined_x, combined_y, combined_confidence, eye_weight, head_weight
    
    def smooth_gaze_coordinates(self, gaze_x, gaze_y, confidence):
        """Apply smoothing to reduce jitter"""
        if confidence > self.config.MIN_CONFIDENCE_THRESHOLD:
            self.smoothed_combined_gaze_x = (
                self.config.SMOOTHING_FACTOR * self.smoothed_combined_gaze_x +
                (1 - self.config.SMOOTHING_FACTOR) * gaze_x
            )
            self.smoothed_combined_gaze_y = (
                self.config.SMOOTHING_FACTOR * self.smoothed_combined_gaze_y +
                (1 - self.config.SMOOTHING_FACTOR) * gaze_y
            )
        
        return self.smoothed_combined_gaze_x, self.smoothed_combined_gaze_y
    
    def process_frame(self, frame):
        """Process a single frame for hybrid gaze detection"""
        results = None
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            rgb_frame.flags.writeable = True
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, 0.5, 0.5, 0.0, {}
        
        gaze_x, gaze_y, confidence = 0.5, 0.5, 0.0
        raw_gaze_x, raw_gaze_y = 0.5, 0.5
        debug_info = {}
        
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Get head pose
                rotation_angles, translation_vec, rotation_mat, translation_vec_3d = \
                    self.head_pose_estimator.estimate_pose(landmarks)
                head_gaze_x, head_gaze_y = self.head_pose_estimator.head_pose_to_gaze_direction(
                    rotation_angles, translation_vec)
                
                # Get eye regions
                left_eye_points = self.get_eye_region(landmarks, self.config.LEFT_EYE_INDICES)
                right_eye_points = self.get_eye_region(landmarks, self.config.RIGHT_EYE_INDICES)
                
                # Get iris centers
                left_iris_center = self.get_iris_center(landmarks, self.config.LEFT_IRIS_INDICES)
                right_iris_center = self.get_iris_center(landmarks, self.config.RIGHT_IRIS_INDICES)
                
                # Calculate eye gaze for both eyes
                left_eye_gaze_x, left_eye_gaze_y, left_conf, left_raw_x, left_raw_y = self.estimate_eye_gaze_direction(
                    left_eye_points, left_iris_center)
                right_eye_gaze_x, right_eye_gaze_y, right_conf, right_raw_x, right_raw_y = self.estimate_eye_gaze_direction(
                    right_eye_points, right_iris_center)
                
                # Average both eyes
                eye_gaze_x = (left_eye_gaze_x + right_eye_gaze_x) / 2
                eye_gaze_y = (left_eye_gaze_y + right_eye_gaze_y) / 2
                eye_confidence = (left_conf + right_conf) / 2
                raw_gaze_x = (left_raw_x + right_raw_x) / 2
                raw_gaze_y = (left_raw_y + right_raw_y) / 2
                
                # Calculate eye closure ratio (for blink detection)
                left_eye_closure = self.calculate_eye_closure_ratio(left_eye_points)
                right_eye_closure = self.calculate_eye_closure_ratio(right_eye_points)
                eye_closure_ratio = (left_eye_closure + right_eye_closure) / 2
                
                # Combine eye gaze and head pose
                combined_x, combined_y, combined_conf, eye_weight, head_weight = \
                    self.combine_gaze_estimates(
                        eye_gaze_x, eye_gaze_y, eye_confidence,
                        head_gaze_x, head_gaze_y, eye_closure_ratio
                    )
                
                # Apply calibration if available
                if self.calibrator.is_calibrated:
                    calibrated_x, calibrated_y = self.calibrator.apply_calibration(combined_x, combined_y)
                    gaze_x, gaze_y = calibrated_x, calibrated_y
                else:
                    gaze_x, gaze_y = combined_x, combined_y
                
                # Apply smoothing
                gaze_x, gaze_y = self.smooth_gaze_coordinates(gaze_x, gaze_y, combined_conf)
                confidence = combined_conf
                
                # Store raw gaze data for calibration
                raw_gaze_data = {
                    'raw_gaze_x': raw_gaze_x,
                    'raw_gaze_y': raw_gaze_y,
                    'eye_features': {
                        'left_eye_closure': left_eye_closure,
                        'right_eye_closure': right_eye_closure,
                        'eye_confidence': eye_confidence
                    },
                    'head_pose': {
                        'rotation_angles': [float(x) for x in rotation_angles],
                        'translation_vec': [float(x) for x in translation_vec.flatten()]
                    }
                }
                
                # Handle calibration data collection
                if self.calibration_collection_active:
                    if self.calibration_mode == CalibrationMode.POINT_CALIBRATION:
                        target_point = self.get_current_calibration_point()
                        self.calibrator.add_calibration_sample(
                            self.current_calibration_point, raw_gaze_data, target_point)
                    elif self.calibration_mode == CalibrationMode.AUTOMATIC_CALIBRATION:
                        # In automatic mode, we'd need some way to know where the user is looking
                        # For now, we'll use the current gaze point as target (self-calibration)
                        target_point = (gaze_x, gaze_y)
                        self.calibrator.add_calibration_sample(0, raw_gaze_data, target_point)
                
                # Store debug information
                debug_info = {
                    'eye_gaze': (float(eye_gaze_x), float(eye_gaze_y)),
                    'head_gaze': (float(head_gaze_x), float(head_gaze_y)),
                    'raw_gaze': (float(raw_gaze_x), float(raw_gaze_y)),
                    'eye_confidence': float(eye_confidence),
                    'eye_closure_ratio': float(eye_closure_ratio),
                    'eye_weight': float(eye_weight),
                    'head_weight': float(head_weight),
                    'head_pose_angles': [float(x) for x in rotation_angles],
                    'combined_before_smoothing': (float(combined_x), float(combined_y)),
                    'calibration_applied': self.calibrator.is_calibrated,
                    'calibration_mode': self.calibration_mode.name
                }
                
                # Convert to screen coordinates for visualization
                screen_x = int(gaze_x * self.config.CAMERA_WIDTH)
                screen_y = int(gaze_y * self.config.CAMERA_HEIGHT)
                
                # Draw visualization
                frame = self.draw_enhanced_visualization(
                    frame, left_eye_points, right_eye_points, 
                    left_iris_center, right_iris_center, 
                    screen_x, screen_y, confidence,
                    rotation_mat, translation_vec_3d,
                    debug_info
                )
        
        # Update calibration state
        if self.calibration_collection_active:
            self.update_calibration()
        
        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return frame, gaze_x, gaze_y, confidence, debug_info
    
    def draw_enhanced_visualization(self, frame, left_eye, right_eye, left_iris, right_iris, 
                                  gaze_x, gaze_y, confidence, rotation_mat, translation_vec, debug_info):
        """Draw enhanced visualization with calibration information"""
        scale_factor = max(1, self.config.CAMERA_WIDTH / 800)
        
        try:
            # Draw coordinate axes for head pose
            axis_length = 50
            axis_points = np.float32([
                [axis_length, 0, 0], 
                [0, axis_length, 0], 
                [0, 0, -axis_length], 
                [0, 0, 0]
            ]).reshape(-1, 3)
            
            img_points, _ = cv2.projectPoints(
                axis_points, rotation_mat, translation_vec,
                self.head_pose_estimator.camera_matrix, 
                self.head_pose_estimator.dist_coeffs
            )
            
            img_points = np.int32(img_points).reshape(-1, 2)
            
            # Draw axes with integer thickness
            origin = tuple(img_points[3])
            cv2.line(frame, origin, tuple(img_points[0]), (0, 0, 255), 2)  # X - Red
            cv2.line(frame, origin, tuple(img_points[1]), (0, 255, 0), 2)  # Y - Green  
            cv2.line(frame, origin, tuple(img_points[2]), (255, 0, 0), 2)  # Z - Blue
            
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
        
        # Draw calibration points if in calibration mode
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION and self.calibration_collection_active:
            # Draw all calibration points
            for i, (px, py) in enumerate(self.config.CALIBRATION_POINTS):
                point_x = int(px * self.config.CAMERA_WIDTH)
                point_y = int(py * self.config.CAMERA_HEIGHT)
                
                if i == self.current_calibration_point:
                    # Current point - draw as animated circle
                    radius = 20 + int(10 * math.sin(time.time() * 5))
                    cv2.circle(frame, (point_x, point_y), radius, (0, 255, 255), 3)
                    cv2.putText(frame, f"LOOK HERE", (point_x - 40, point_y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Other points
                    cv2.circle(frame, (point_x, point_y), 15, (100, 100, 100), 2)
                
                cv2.putText(frame, str(i+1), (point_x - 5, point_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw gaze point
        gaze_size = 8
        color = (0, 255, 0) if self.calibrator.is_calibrated else (255, 0, 255)
        cv2.circle(frame, (gaze_x, gaze_y), gaze_size, color, -1)
        cv2.circle(frame, (gaze_x, gaze_y), gaze_size + 3, (255, 255, 255), 2)
        
        # Draw eye points
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Draw iris centers
        if left_iris:
            cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 0, 255), -1)
        if right_iris:
            cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 0, 255), -1)
        
        # Get system stats
        system_stats = self.system_monitor.get_system_stats()
        
        # Draw information text
        font_scale = 0.5
        thickness = 1
        y_offset = 30
        
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Confidence: {confidence:.2f}",
            f"Gaze: ({gaze_x/self.config.CAMERA_WIDTH:.2f}, {gaze_y/self.config.CAMERA_HEIGHT:.2f})",
            f"Calibrated: {self.calibrator.is_calibrated}",
            f"Calibration Mode: {self.calibration_mode.name}",
            f"CPU: {system_stats['cpu_percent']:.1f}%",
            f"Memory: {system_stats['memory_used_mb']:.0f}MB ({system_stats['memory_percent']:.1f}%)"
        ]
        
        if self.calibration_mode == CalibrationMode.POINT_CALIBRATION and self.calibration_collection_active:
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.config.CALIBRATION_DURATION - elapsed)
            info_lines.append(f"Calibration Point: {self.current_calibration_point + 1}/{len(self.config.CALIBRATION_POINTS)}")
            info_lines.append(f"Time remaining: {remaining:.1f}s")
        
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * 20
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), thickness + 1)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 0, 0), thickness)
        
        # Draw head pose angles if available
        if 'head_pose_angles' in debug_info:
            angles = debug_info['head_pose_angles']
            pose_text = f"Head: P{np.degrees(angles[0]):.1f} Y{np.degrees(angles[1]):.1f} R{np.degrees(angles[2]):.1f}"
            y_pos = y_offset + len(info_lines) * 20
            cv2.putText(frame, pose_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), thickness + 1)
            cv2.putText(frame, pose_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 0), thickness)
        
        return frame
    
    def get_gaze_data(self, gaze_x, gaze_y, confidence, debug_info):
        """Format gaze data for output with system monitoring"""
        timestamp = datetime.now().isoformat()
        
        # Get system statistics
        system_stats = self.system_monitor.get_system_stats()
        
        data = {
            "timestamp": timestamp,
            "gaze_x": float(gaze_x / self.config.CAMERA_WIDTH),
            "gaze_y": float(gaze_y / self.config.CAMERA_HEIGHT),
            "confidence": float(confidence),
            "fps": float(self.fps),
            "calibrated": self.calibrator.is_calibrated,
            "eye_gaze_x": float(debug_info.get('eye_gaze', (0, 0))[0]),
            "eye_gaze_y": float(debug_info.get('eye_gaze', (0, 0))[1]),
            "head_gaze_x": float(debug_info.get('head_gaze', (0, 0))[0]),
            "head_gaze_y": float(debug_info.get('head_gaze', (0, 0))[1]),
            "raw_gaze_x": float(debug_info.get('raw_gaze', (0, 0))[0]),
            "raw_gaze_y": float(debug_info.get('raw_gaze', (0, 0))[1]),
            "eye_confidence": float(debug_info.get('eye_confidence', 0)),
            "eye_closure_ratio": float(debug_info.get('eye_closure_ratio', 0)),
            "eye_weight": float(debug_info.get('eye_weight', 0)),
            "head_weight": float(debug_info.get('head_weight', 0)),
            "head_pose_angles": debug_info.get('head_pose_angles', [0, 0, 0]),
            "resolution": f"{self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}",
            "cpu_percent": float(system_stats['cpu_percent']),
            "memory_used_mb": float(system_stats['memory_used_mb']),
            "memory_percent": float(system_stats['memory_percent'])
        }
        
        self.gaze_data.append(data)
        
        # Write to CSV if enabled
        self.write_csv_row(data)
        
        return data

# Enhanced Data Export Functions with System Monitoring
def save_gaze_data_to_csv(gaze_data, filename=None):
    """Save gaze data to CSV file with comprehensive formatting"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaze_data_export_{timestamp}.csv"
    
    try:
        # Create DataFrame
        df = pd.DataFrame(gaze_data)
        
        # Convert head pose angles to degrees for better readability
        if 'head_pose_angles' in df.columns:
            df['pitch_deg'] = df['head_pose_angles'].apply(lambda x: np.degrees(x[0]) if x else 0)
            df['yaw_deg'] = df['head_pose_angles'].apply(lambda x: np.degrees(x[1]) if x else 0)
            df['roll_deg'] = df['head_pose_angles'].apply(lambda x: np.degrees(x[2]) if x else 0)
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Gaze data exported to CSV: {filename}")
        print(f"Total records: {len(df)}")
        
        # Print summary statistics including system metrics
        if len(df) > 0:
            print("\nSummary Statistics:")
            print(f"Average confidence: {df['confidence'].mean():.3f}")
            print(f"Average FPS: {df['fps'].mean():.1f}")
            print(f"Gaze X range: [{df['gaze_x'].min():.3f}, {df['gaze_x'].max():.3f}]")
            print(f"Gaze Y range: [{df['gaze_y'].min():.3f}, {df['gaze_y'].max():.3f}]")
            print(f"Average CPU usage: {df['cpu_percent'].mean():.1f}%")
            print(f"Average Memory usage: {df['memory_percent'].mean():.1f}%")
            print(f"Peak Memory usage: {df['memory_used_mb'].max():.0f}MB")
            
        return filename
        
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return None

def save_gaze_data_to_json(gaze_data, filename=None):
    """Save gaze data to JSON file with system monitoring data"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaze_data_{timestamp}.json"
    
    try:
        # Add system summary to JSON data
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_samples": len(gaze_data),
                "system_summary": {
                    "avg_cpu_percent": np.mean([d.get('cpu_percent', 0) for d in gaze_data]),
                    "avg_memory_percent": np.mean([d.get('memory_percent', 0) for d in gaze_data]),
                    "max_memory_used_mb": max([d.get('memory_used_mb', 0) for d in gaze_data]),
                    "avg_fps": np.mean([d.get('fps', 0) for d in gaze_data]),
                    "avg_confidence": np.mean([d.get('confidence', 0) for d in gaze_data])
                }
            },
            "gaze_data": gaze_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Gaze data saved to JSON: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving JSON data: {e}")
        return None

def export_calibration_report(gaze_tracker, filename=None):
    """Export comprehensive calibration report with system metrics"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_report_{timestamp}.txt"
    
    try:
        # Calculate system statistics from gaze data
        if gaze_tracker.gaze_data:
            cpu_stats = {
                'avg': np.mean([d.get('cpu_percent', 0) for d in gaze_tracker.gaze_data]),
                'max': max([d.get('cpu_percent', 0) for d in gaze_tracker.gaze_data]),
                'min': min([d.get('cpu_percent', 0) for d in gaze_tracker.gaze_data])
            }
            memory_stats = {
                'avg_mb': np.mean([d.get('memory_used_mb', 0) for d in gaze_tracker.gaze_data]),
                'max_mb': max([d.get('memory_used_mb', 0) for d in gaze_tracker.gaze_data]),
                'avg_percent': np.mean([d.get('memory_percent', 0) for d in gaze_tracker.gaze_data])
            }
        else:
            cpu_stats = {'avg': 0, 'max': 0, 'min': 0}
            memory_stats = {'avg_mb': 0, 'max_mb': 0, 'avg_percent': 0}
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Gaze Tracking Calibration Report ===\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Camera Resolution: {gaze_tracker.config.CAMERA_WIDTH}x{gaze_tracker.config.CAMERA_HEIGHT}\n")
            f.write(f"Calibration Status: {gaze_tracker.calibrator.is_calibrated}\n")
            f.write(f"Total Gaze Samples: {len(gaze_tracker.gaze_data)}\n")
            f.write(f"Average FPS: {gaze_tracker.fps:.1f}\n")
            
            f.write("\nSystem Performance Metrics:\n")
            f.write(f"  Average CPU Usage: {cpu_stats['avg']:.1f}%\n")
            f.write(f"  Maximum CPU Usage: {cpu_stats['max']:.1f}%\n")
            f.write(f"  Average Memory Usage: {memory_stats['avg_mb']:.0f}MB ({memory_stats['avg_percent']:.1f}%)\n")
            f.write(f"  Peak Memory Usage: {memory_stats['max_mb']:.0f}MB\n")
            
            f.write("\nCalibration Details:\n")
            if hasattr(gaze_tracker.calibrator, 'calibration_samples'):
                for point_idx, samples in gaze_tracker.calibrator.calibration_samples.items():
                    f.write(f"  Point {point_idx}: {len(samples)} samples\n")
            
            f.write("\nSystem Configuration:\n")
            f.write(f"  Eye Gaze Weight: {gaze_tracker.config.EYE_GAZE_WEIGHT}\n")
            f.write(f"  Head Gaze Weight: {gaze_tracker.config.HEAD_GAZE_WEIGHT}\n")
            f.write(f"  Smoothing Factor: {gaze_tracker.config.SMOOTHING_FACTOR}\n")
            f.write(f"  Min Confidence: {gaze_tracker.config.MIN_CONFIDENCE_THRESHOLD}\n")
            f.write(f"  System Monitor Interval: {gaze_tracker.config.SYSTEM_MONITOR_INTERVAL}s\n")
        
        print(f"Calibration report saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving calibration report: {e}")
        return None

# Enhanced Main Function with Comprehensive Logging and System Monitoring
def run_hybrid_gaze_tracking_with_calibration():
    """Main function to run hybrid gaze tracking with calibration"""
    
    config = GazeConfig()
    
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    config.CAMERA_WIDTH = actual_width
    config.CAMERA_HEIGHT = actual_height
    
    print(f"Camera: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize hybrid tracker
    gaze_tracker = HybridGazeTracker(config)
    
    # Setup CSV logging
    gaze_tracker.setup_csv_logging()
    
    print("Starting hybrid gaze tracking with calibration and system monitoring...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save data (JSON + CSV)")
    print("  'c' - Start point calibration")
    print("  'a' - Start automatic calibration")
    print("  'l' - Load calibration")
    print("  'r' - Reset calibration")
    print("  'e' - Export calibration report")
    print("  'm' - Show system metrics")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame with hybrid tracking
            processed_frame, gaze_x, gaze_y, confidence, debug_info = gaze_tracker.process_frame(frame)
            
            # Get gaze data
            gaze_data = gaze_tracker.get_gaze_data(gaze_x, gaze_y, confidence, debug_info)
            
            # Display
            cv2.imshow('Hybrid Gaze Tracking with Calibration', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save data in multiple formats
                json_file = save_gaze_data_to_json(gaze_tracker.gaze_data)
                csv_file = save_gaze_data_to_csv(gaze_tracker.gaze_data)
                print(f"Data saved: {len(gaze_tracker.gaze_data)} samples")
            elif key == ord('c'):
                gaze_tracker.start_calibration("point")
                print("Starting point calibration...")
            elif key == ord('a'):
                gaze_tracker.start_calibration("auto")
                print("Starting automatic calibration...")
            elif key == ord('l'):
                # Load calibration from file
                gaze_tracker.calibrator.load_calibration("gaze_calibration_latest.json")
            elif key == ord('r'):
                gaze_tracker.calibrator.is_calibrated = False
                gaze_tracker.calibrator.calibration_model = None
                print("Calibration reset")
            elif key == ord('e'):
                export_calibration_report(gaze_tracker)
            elif key == ord('m'):
                # Show current system metrics
                stats = gaze_tracker.system_monitor.get_system_stats()
                print(f"\nCurrent System Metrics:")
                print(f"  CPU: {stats['cpu_percent']:.1f}%")
                print(f"  Memory: {stats['memory_used_mb']:.0f}MB ({stats['memory_percent']:.1f}%)")
                print(f"  FPS: {gaze_tracker.fps:.1f}")
                print(f"  Confidence: {confidence:.3f}")
    
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop system monitoring
        gaze_tracker.system_monitor.stop_monitoring()
        
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(gaze_tracker, 'face_mesh'):
            gaze_tracker.face_mesh.close()
        
        # Close CSV logging
        gaze_tracker.close_csv_logging()
        
        # Save final data
        if gaze_tracker.gaze_data:
            json_file = save_gaze_data_to_json(gaze_tracker.gaze_data)
            csv_file = save_gaze_data_to_csv(gaze_tracker.gaze_data)
            report_file = export_calibration_report(gaze_tracker)
            print(f"Final data saved: {len(gaze_tracker.gaze_data)} samples")

# Run the enhanced hybrid gaze tracking system with calibration and system monitoring
if __name__ == "__main__":
    run_hybrid_gaze_tracking_with_calibration()
