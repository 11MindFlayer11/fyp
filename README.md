# Vision-Based Roof Gantry Crane System

Real-time computer-vision pipeline for a single-camera, ArUco-guided gantry crane that detects steel pipes on the floor and commands the crane to pick them up.

```
Frame → Undistort → ArUco Pose → YOLO Detect → Pixel→World → Track → Select → Motor Command
```

---

## Project Structure

```
crane_vision/
├── main.py                        ← Entry point (real-time loop)
├── requirements.txt
│
├── config/
│   └── settings.py                ← ALL tunable parameters
│
├── calibration/
│   ├── calibrate_camera.py        ← Offline checkerboard calibration
│   └── images/                    ← Store calibration frames here
│
├── localization/
│   └── aruco_pose.py              ← ArUco detection + solvePnP pose
│
├── detection/
│   └── pipe_detector.py           ← YOLOv8 pipe detection
│
├── tracking/
│   ├── pipe_tracker.py            ← IoU tracker + moving-average smoothing
│   └── target_selector.py         ← Pick which pipe to grab
│
├── control/
│   ├── crane_controller.py        ← Serial commands to Arduino
│   └── crane_controller.ino       ← Arduino firmware (AccelStepper)
│
├── utils/
│   └── helpers.py                 ← FPS counter, HUD overlay, bounds check
│
├── models/                        ← Place YOLO .pt weights here
├── logs/
├── output/
└── tests/
    └── test_pipeline.py
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU acceleration** (optional):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

---

### 2. Camera calibration (one-time)

Print a **9×6 checkerboard** (inner corners) with 25 mm squares.

```bash
# Step A: collect calibration frames interactively
python -m calibration.calibrate_camera --collect --count 30

# Step B: compute intrinsics
python -m calibration.calibrate_camera --calibrate

# Or both at once:
python -m calibration.calibrate_camera --full
```

Target RMS reprojection error: **< 1.0 px**.
Result saved to `config/camera_calibration.npz`.

---

### 3. Print and place ArUco markers

```bash
python main.py --generate-markers
# → output/markers/marker_00.png … marker_07.png
```

Print each at the physical size set by `ARUCO_MARKER_SIZE_M` (default 12 cm).
Place them flat on the floor and **measure their centre positions** in your world frame.
Update `ARUCO_WORLD_POSITIONS` in `config/settings.py`.

**Tips for marker placement:**
- Spread across the full workspace (corners + centre)
- No three markers should be collinear
- Keep markers flat and unobstructed

---

### 4. Configure `config/settings.py`

Key parameters to verify before first run:

| Parameter | Description |
|---|---|
| `CAMERA_INDEX` | OpenCV camera index |
| `FRAME_WIDTH / HEIGHT` | Capture resolution |
| `ARUCO_WORLD_POSITIONS` | Measured floor positions of your markers |
| `ARUCO_MARKER_SIZE_M` | Printed marker side length in metres |
| `YOLO_MODEL_PATH` | Path to your trained YOLO weights |
| `YOLO_PIPE_CLASS_ID` | Class ID of "pipe" in your model |
| `WORKSPACE_*` | Crane working area bounds in metres |
| `SERIAL_PORT` | `/dev/ttyUSB0` (Linux) or `COM3` (Windows) |
| `SERIAL_ENABLED` | `False` for testing without hardware |
| `CRANE_HOME_POSITION` | Used by "closest" target selection |

---

### 5. YOLO model

The system ships with `yolov8n.pt` (nano) as default. For best accuracy:

```bash
# Option A: Use a pre-trained COCO model (if pipes look like known objects)
# Option B: Train on your own pipe images
yolo train data=pipe_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Place weights at the path set by `YOLO_MODEL_PATH`.

---

### 6. Run the system

```bash
# Normal operation (serial disabled by default in settings)
python main.py

# With hardware connected
# First set SERIAL_ENABLED = True in settings.py, then:
python main.py

# Force serial off regardless of settings
python main.py --dry-run

# Headless (server / Raspberry Pi)
python main.py --no-display

# Record annotated video
python main.py --record output/run_001.mp4
```

**Window controls:**
- `Q` — quit
- `R` — reset tracker

---

### 7. Flash Arduino

Open `control/crane_controller.ino` in the Arduino IDE.
Install the **AccelStepper** library (Library Manager).
Adjust pin numbers and `STEPS_PER_METRE_*` to match your hardware, then upload.

---

## How It Works

### Coordinate System

```
World frame (metres):
  Origin  = Marker 0 centre
  +X      = right along floor
  +Y      = forward along floor
  +Z      = up (away from floor)
  Floor   = Z = 0
```

### Pixel → World (single camera, known floor)

Since pipes lie on a flat floor (Z = 0), the 3D point is fully determined by the camera ray and the floor plane intersection:

```
ray_cam   = K⁻¹ · [px, py, 1]ᵀ
ray_world = R_cam_to_world · ray_cam
origin    = camera centre in world

t = -origin.z / ray_world.z
point = origin + t · ray_world   →  (X, Y, 0) on floor
```

Z is then offset by `PIPE_RADIUS_M` so the crane grabs the pipe centre.

### Stability Gate

A pipe command is only sent once it has been seen with consistent coordinates for `STABILITY_FRAMES` consecutive frames (XY std-dev < `STABILITY_TOLERANCE_M`). This prevents jitter from triggering false pickups.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "No ArUco markers detected" | Bad lighting / markers too small | Improve lighting; increase marker size |
| Large world coordinate errors | Marker positions wrong in config | Re-measure and update `ARUCO_WORLD_POSITIONS` |
| High reprojection RMS (> 2px) | Poor calibration | Collect more diverse calibration frames |
| YOLO detects nothing | Wrong class ID or confidence too high | Lower `YOLO_CONFIDENCE`; verify `YOLO_PIPE_CLASS_ID` |
| Serial timeout | Arduino not responding | Check baud rate; re-flash firmware |
| Coordinates jump frame-to-frame | Camera vibration or low FPS | Mount camera rigidly; increase `SMOOTHING_WINDOW` |

---

## Future Improvements

- [ ] **Kalman filter** (replace moving average for better prediction)
- [ ] **Custom YOLO fine-tune** on pipe dataset
- [ ] **Closed-loop feedback** via encoder readings from crane
- [ ] **Multi-marker PnP** with outlier rejection (already partially done via RANSAC)
- [ ] **Simulation mode** with synthetic frames for testing without hardware
- [ ] **Web dashboard** for remote monitoring

---

## License

MIT — adapt freely for your robotic application.
