"""
config/settings.py
Central configuration for the Vision-Based Roof Gantry Crane System.
Edit this file to match your hardware setup.
"""

import numpy as np

# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
CAMERA_INDEX       = "https://172.28.36.4:8080/video"         # OpenCV camera index (0 = default webcam)
FRAME_WIDTH        = 1280       # Capture resolution width  (px)
FRAME_HEIGHT       = 720        # Capture resolution height (px)
CAMERA_FPS         = 30         # Target frames per second
CAMERA_TILT_DEG    = 45.0       # Approximate tilt angle from vertical (degrees)
                                # Used only as metadata / sanity check

# ─────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────
CALIBRATION_FILE       = "config/camera_calibration.npz"
CHECKERBOARD_SIZE      = (9, 6)     # Inner corners (cols, rows)
CHECKERBOARD_SQUARE_MM = 25.0       # Physical size of one square in mm
CALIBRATION_IMAGES_DIR = "calibration/images/"
MIN_CALIBRATION_IMAGES = 15         # Minimum valid frames for calibration

# ─────────────────────────────────────────────
#  ARUCO MARKERS
# ─────────────────────────────────────────────
ARUCO_DICT         = "DICT_5X5_250"   # ArUco dictionary name
ARUCO_MARKER_SIZE_M = 0.12           # Physical marker side length in CMS
ARUCO_MARKER_IDS   = [0, 3, 4, 5]   # Expected marker IDs
MIN_MARKERS_FOR_POSE = 3             # Minimum visible markers for valid pose

# Known world coordinates of ArUco marker centres (in METRES).
# Origin (0,0,0) = your chosen reference point on the floor.
# Update these to match your actual floor layout.
ARUCO_WORLD_POSITIONS = {
    0: np.array([.0825, .0825, 0.0]),
    # 1: np.array([1.00,  0.00, 0.0]),
    # 2: np.array([2.00,  0.00, 0.0]),
    3: np.array([.0825, .4675, 0.0]),
    4: np.array([.6175, .4675, 0.0]),
    5: np.array([.6175, .0825, 0.0]),
    # 6: np.array([0.50,  1.50, 0.0]),
    # 7: np.array([1.50,  1.50, 0.0]),
}

# ─────────────────────────────────────────────
#  YOLO PIPE DETECTION
# ─────────────────────────────────────────────
YOLO_MODEL_PATH    = "models/yolov8n.pt"      # Path to model weights
YOLO_CONFIDENCE    = 0.45                     # Detection confidence threshold
YOLO_IOU_THRESHOLD = 0.45                     # NMS IoU threshold
YOLO_DEVICE        = "cpu"                    # "cpu" | "cuda" | "mps"
YOLO_PIPE_CLASS_ID = None                       # Class ID for "pipe" in your model
                                              # Set to None to use all detections
YOLO_INPUT_SIZE    = 640                      # YOLO inference resolution

# ─────────────────────────────────────────────
#  WORKSPACE / CRANE BOUNDS  (metres)
# ─────────────────────────────────────────────
WORKSPACE_X_MIN    = 0.0
WORKSPACE_X_MAX    = 0.7
WORKSPACE_Y_MIN    = 0.0
WORKSPACE_Y_MAX    = 0.55
WORKSPACE_Z_MIN    = -0.05   # Allow slight below-floor tolerance
WORKSPACE_Z_MAX    =  0.20   # Max height of a pipe lying on the floor

# ─────────────────────────────────────────────
#  PIPE GEOMETRY
# ─────────────────────────────────────────────
PIPE_RADIUS_M      = 0.025   # Approximate pipe radius in metres
                              # Added as Z-offset so crane grabs centre of pipe

# ─────────────────────────────────────────────
#  TRACKING / SMOOTHING
# ─────────────────────────────────────────────
SMOOTHING_WINDOW       = 5    # Frames for moving-average smoothing
STABILITY_FRAMES       = 8    # Consecutive stable frames before pickup command
STABILITY_TOLERANCE_M  = 0.02 # Max position change (m) to be considered "stable"
MAX_TRACKED_PIPES      = 10   # Maximum number of simultaneously tracked pipes

# ─────────────────────────────────────────────
#  TARGET SELECTION
# ─────────────────────────────────────────────
# Strategy: "closest" | "first_detected" | "manual"
TARGET_SELECTION_STRATEGY = "closest"
CRANE_HOME_POSITION = np.array([1.0, 0.75, 0.5])   # metres, used for "closest"

# ─────────────────────────────────────────────
#  SERIAL / CRANE CONTROL
# ─────────────────────────────────────────────
SERIAL_PORT        = "/dev/ttyUSB0"   # Linux: /dev/ttyUSB0  Windows: COM3
SERIAL_BAUD        = 115200
SERIAL_TIMEOUT     = 1.0              # seconds
SERIAL_ENABLED     = False            # Set True to send commands to hardware
COMMAND_FORMAT     = "{x:.4f},{y:.4f},{z:.4f}\n"

# ─────────────────────────────────────────────
#  LOGGING / OUTPUT
# ─────────────────────────────────────────────
LOG_DIR            = "logs/"
OUTPUT_DIR         = "output/"
SAVE_ANNOTATED_VIDEO = False
ANNOTATED_VIDEO_PATH = "output/annotated.mp4"
DISPLAY_WINDOW     = True
PRINT_COORDS       = True
LOG_LEVEL          = "INFO"    # DEBUG | INFO | WARNING | ERROR
