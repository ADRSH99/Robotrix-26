# Robotrix 2025-26 Final Hackathon — Approach Documentation

**Author:** Aniruddh  
**Team:** Aniruddh, Adarsh

---

## Problem Summary

### Objective

Guide a blind rover ("The Mole") through a 15m × 15m maze using only visual feedback from a drone ("The Hawk") flying at low altitude. The drone's camera provides a limited 512×512 view of the maze floor.

### Constraints

| Constraint | Description |
|------------|-------------|
| No Global Position | `sim.getObjectPosition()` banned |
| No State Sharing | No global variables between drone and rover |
| Velocity-Only Control | Only velocity commands can be sent to the Mole |
| Limited FOV | Camera sees only local maze section |

---

## Solution Approach

Our solution implements a **Multi-State Visual Wall Follower** with these key components:

### 1. Mole Detection (Yellow Marker Tracking)
- HSV color thresholding to detect yellow marker on mole
- Morphological cleanup (open/close) to remove noise
- Contour detection + minimum enclosing circle for robust position

### 2. Drone Tracking (PID Controller)
- Keeps mole centered in camera frame using PID control
- Separate X/Y axis PID with anti-windup clamping
- Smooth pursuit allows mole to be consistently visible

### 3. Wall Detection (Red Wall Segmentation)
- Two-range HSV detection for red (hue wraps around 0/180)
- Morphological closing to fill gaps in wall masks

### 4. Guide Rail Generation (Virtual Line Following)
- **Closest-point wall selection** instead of centroid-based
- Samples a point 100px to the mole's RIGHT side
- Selects wall contour closest to that sample point
- Prevents wall-jumping at corners during turns
- Morphological fillet (dilate + erode) creates smooth offset path
- `GUIDE_RAIL_OFFSET = 70px` from wall surface
- `CURVE_RADIUS = 40px` for smooth corner rounding

### 5. 5-Sensor Virtual IR Array
- Reverse-U arrangement: `FAR_L`, `LEFT`, `CENTER`, `RIGHT`, `FAR_R`
- Weighted error calculation for PID steering
- Sensor positions calculated based on mole's estimated heading

### 6. Heading Estimation (Hybrid Sensor Fusion)
- **Primary:** Wheel odometry from velocity commands
- **Secondary:** Visual heading from guide rail using `cv2.fitLine`
- Complementary filter: **98% odometry + 2% vision**
- Eliminates long-term drift while maintaining smooth motion

### 7. State Machine Navigation
- `WALL_FOLLOW`: Primary state, right-hand wall following
- `LOOP_ESCAPE`: Escape behavior when stuck (rotate in place)

### 8. Velocity Ramping
- Smooth acceleration/deceleration prevents jerky motion
- `ACCEL_STEP = 0.5` units per frame for gradual speed changes

---

## PID Tuning Parameters

### Drone Tracking
```
DRONE_KP = 0.0058
DRONE_KI = 0.00018 (with anti-windup)
DRONE_KD = 0.0175
```

### Line Following
```
LINE_FOLLOW_KP = 0.0152
LINE_FOLLOW_KI = 0.00015 (minimal to prevent windup)
LINE_FOLLOW_KD = 0.025
BASE_SPEED = 2.0 units
```

### 5-Sensor Weights

| Sensor | Weight | Effect |
|--------|--------|--------|
| FAR_L | +2.3 | Strong left turn |
| LEFT | +1.2 | Mild left turn |
| CENTER | 0.0 | No correction |
| RIGHT | -1.2 | Mild right turn |
| FAR_R | -2.3 | Strong right turn |

---

## Key Design Decisions

### 1. Closest-Point vs Centroid Wall Selection

**Problem:** Centroid-based selection caused wall-jumping at corners

**Solution:** Sample a point to the RIGHT of the mole and find the wall contour closest to that point. This provides stable transitions.

### 2. Virtual 5-Sensor Array vs Simple Error

**Problem:** Single-point tracking lost the line easily

**Solution:** 5 virtual IR sensors in reverse-U layout with weighted error provides robust line detection even during oscillations.

### 3. Hybrid Heading Estimation

**Problem:** Wheel odometry drifts over time

**Solution:** Fuse with visual heading from fitted line at 2% weight. High odometry weight (98%) ensures smooth motion, low vision weight kills long-term drift.

### 4. Morphological Fillet

**Problem:** Sharp corners in guide rail caused overshooting

**Solution:** Dilate + erode creates naturally rounded corners that the virtual sensors can follow smoothly.

---

## File Structure

```
task_submission.py
├── Global Constants (HSV thresholds, PID gains, sensor config)
├── Path Detection Module
│   ├── detect_green_path()
│   ├── get_right_green_contour()
│   └── generate_green_guide_rail()
├── Wall Following Module
│   ├── get_right_wall_contour()
│   └── generate_guide_rail()
├── Virtual Sensor Module
│   ├── read_virtual_sensors()
│   └── compute_line_steering()
├── Heading Estimation
│   └── calculate_visual_heading()
├── Core Functions (provided)
│   ├── detect_mole()
│   ├── compute_drone_velocity()
│   ├── detect_walls()
│   ├── get_camera_image()
│   └── send_mole_velocity() / send_hawk_velocity()
└── control_logic() — Main state machine and control loop
```

---

## Visualization

Two OpenCV windows are displayed during execution:

1. **"Camera"** — Raw camera view with mole detection circle and state label
2. **"Guide Rail"** — Wall mask overlay with:
   - Yellow guide rail path
   - Green mole marker with cyan heading arrow
   - 5 colored sensor boxes (blue-to-red gradient)
   - Current heading value

---

## Results

🏆 **1st Place — IEEE Robotrix 2025-26 Final Hackathon**

The approach successfully navigated the complete maze in real-time, demonstrating robust wall following through corners, T-junctions, and varying corridor widths.

---

## Acknowledgments

This solution uses:
- CoppeliaSim ZeroMQ Remote API
- OpenCV for computer vision
- NumPy for numerical operations
- Standard Python math library
