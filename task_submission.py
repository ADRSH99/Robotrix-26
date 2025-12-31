'''
*****************************************************************************************
*
*        =================================================
*                   IEEE Robotrix 2025-26
*        =================================================
*
*  This script is intended for implementation of the final hackathon
*  task of IEEE Robotrix 2025-26
*
*****************************************************************************************
'''

# Team Name:
# Team Members:
# Filename:            task.py
# Functions:           control_logic, get_camera_image, send_mole_velocity, send_hawk_velocity
# Global variables:    NONE (Only allowed if strictly required)

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import sys
import traceback
import time
import os
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import random
##############################################################

################# ADD GLOBAL VARIABLES HERE ##################

# STEP 1: MOLE DETECTION - HSV Color Thresholds
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

#STEP 2: DRONE TRACKING
IMAGE_CENTER = (256, 256)  # Center of 512x512 image
DRONE_CENTER_OFFSET = (0, 10)  # Calibration offset (x, y) - tune if mole drifts

# PID Constants
DRONE_KP = 0.0058   
DRONE_KI = 0.00018  # Slightly increased to eliminate steady-state drift
DRONE_KD = 0.0175   

# PID state
pid_state = {
    'integral_x': 0.0,
    'integral_y': 0.0,
    'prev_error_x': 0.0,
    'prev_error_y': 0.0
}

# STEP 3: WALL DETECTION - HSV Thresholds
# Walls are RED - need two ranges since red wraps around in HSV
WALL_LOWER_1 = np.array([0, 100, 100])    # Red range 1: Hue 0-10
WALL_UPPER_1 = np.array([10, 255, 255])
WALL_LOWER_2 = np.array([170, 100, 100])  # Red range 2: Hue 170-180
WALL_UPPER_2 = np.array([180, 255, 255])

# Heading tracking state
heading_state = {
    'heading': 0.0,        # Degrees, 0=Up, 90=Right, 180=Down, 270=Left
    'last_time': None,
    'last_vl': 0.0,
    'last_vr': 0.0,
}
ANGULAR_RATE = 29.0  # degrees per second per unit velocity difference

# GREEN PATH DETECTION - For line following
GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([85, 255, 255])
GREEN_OFFSET = 30  # Smaller offset for green path (tighter tracking)

# GOAL DETECTION - Large green blob at end
GOAL_MIN_AREA = 3000  # Minimum area to be considered goal

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

##############################################################


#LINE FOLLOWER MODULE - 5 IR SENSORS

# Configuration
GUIDE_RAIL_OFFSET = 70      # Pixels offset from wall
CURVE_RADIUS = 40           # Radius for morphological fillet (smooth corners)

# 5-Sensor Array Configuration (Reverse-U Layout)
# Each sensor: (forward_offset, lateral_offset, weight)
# Positive lateral = RIGHT, Negative lateral = LEFT
# Positive weight = turns LEFT when triggered, Negative = turns RIGHT
SENSOR_ARRAY = [
    {'name': 'FAR_L',  'forward': 45, 'lateral': -35, 'weight':  2.3},
    {'name': 'LEFT',   'forward': 55, 'lateral': -15, 'weight':  1.2},
    {'name': 'CENTER', 'forward': 60, 'lateral':   0, 'weight':  0.0},
    {'name': 'RIGHT',  'forward': 55, 'lateral':  15, 'weight': -1.2},
    {'name': 'FAR_R',  'forward': 45, 'lateral':  35, 'weight': -2.3},
]
SENSOR_SIZE = 10            # Size of sensor ROI (square)

# PID Controller Constants
LINE_FOLLOW_KP = 0.0152      # Proportional gain (increased for faster response)
LINE_FOLLOW_KI = 0.00015     # Integral gain (minimal to zero windup)
LINE_FOLLOW_KD = 0.025      # Derivative gain (high damping for stability)
LINE_FOLLOW_BASE_SPEED = 2.0


# Path Detection Module

def detect_green_path(image):
    """Detect green pixels in the image (the intended path to follow)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    return green_mask


def get_right_green_contour(contours, mole_pos, heading):
    """
    Select the green contour that is to the RIGHT of the robot using cross-product.
    This is more stable than polygon masking.
    
    Returns: The best right-side contour, or closest as fallback
    """
    if mole_pos is None or len(contours) == 0:
        return None
    
    cx, cy = mole_pos
    
    # Calculate heading vector (0=Up, 90=Right)
    heading_rad = math.radians(heading)
    # Forward vector
    fwd_x = math.sin(heading_rad)
    fwd_y = -math.cos(heading_rad)
    # Right vector (perpendicular to forward, 90° clockwise)
    right_x = -fwd_y  # Perpendicular
    right_y = fwd_x
    
    best_contour = None
    best_score = -float('inf')
    closest_contour = None
    closest_distance = float('inf')
    max_distance = 450  # Increased to detect paths farther away
    
    for contour in contours:
        # Skip tiny contours
        if cv2.contourArea(contour) < 50:
            continue
            
        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        
        # Vector from mole to centroid
        vx = centroid_x - cx
        vy = centroid_y - cy
        
        # Distance check
        distance = math.sqrt(vx*vx + vy*vy)
        if distance > max_distance:
            continue
        
        # Track closest as fallback
        if distance < closest_distance:
            closest_distance = distance
            closest_contour = contour
        
        # Dot product with RIGHT vector
        # Positive = path is to the RIGHT of mole
        dot_right = vx * right_x + vy * right_y
        
        # Score: prioritize paths that are CLOSE, then use right-alignment as tie breaker
        if dot_right > 0:  # Positive = right side
            # New Score: Heavy penalty on distance, light bonus for right-alignment
            score = -distance * 5.0 + dot_right * 1.0
            if score > best_score:
                best_score = score
                best_contour = contour
    
    # Return best right-side path, or fallback to closest
    if best_contour is not None:
        return best_contour
    else:
        return closest_contour


def generate_green_guide_rail(green_mask, mole_pos, heading):
    """
    Generate a guide rail from the green path using vector-based contour selection.
    
    Returns: (guide_rail, goal_detected)
    """
    if mole_pos is None:
        return np.zeros_like(green_mask), False
    
    # 1. Find all green contours
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(green_contours) == 0:
        return np.zeros_like(green_mask), False
    
    # Check for goal (large green blob)
    goal_detected = False
    for contour in green_contours:
        if cv2.contourArea(contour) > GOAL_MIN_AREA:
            goal_detected = True
            break
    
    # 2. Select the RIGHT green path using cross-product
    right_green = get_right_green_contour(green_contours, mole_pos, heading)
    
    if right_green is None:
        return np.zeros_like(green_mask), goal_detected
    
    # 3. Create mask with ONLY the selected green path
    selected_mask = np.zeros_like(green_mask)
    cv2.drawContours(selected_mask, [right_green], -1, 255, cv2.FILLED)
    
    # 4. Dilate to create guide rail (offset from path edge)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GREEN_OFFSET*2, GREEN_OFFSET*2))
    dilated = cv2.dilate(selected_mask, dilate_kernel, iterations=1)
    
    # 5. Extract the edge as the guide rail
    guide_rail = np.zeros_like(green_mask)
    path_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw thicker line for better sensor detection
    cv2.drawContours(guide_rail, path_contours, -1, 255, 9)
    
    return guide_rail, goal_detected


# Wall Based Guide Rail (Fallback)

def get_right_wall_contour(contours, mole_pos, heading):
    """
    Select the wall contour that is to the RIGHT of the robot using cross-product.
    
    NOTE: In image coordinates, Y increases DOWNWARD, so cross-product sign is inverted.
    
    Returns: The best right-side contour, or closest wall as fallback
    """
    if mole_pos is None or len(contours) == 0:
        return None
    
    cx, cy = mole_pos
    
    # Calculate heading vector (0=Up, 90=Right)
    heading_rad = math.radians(heading)
    # Forward vector
    fwd_x = math.sin(heading_rad)
    fwd_y = -math.cos(heading_rad)
    # Right vector (perpendicular to forward, 90° clockwise)
    right_x = -fwd_y
    right_y = fwd_x
    
    best_contour = None
    best_score = -float('inf')
    closest_contour = None
    closest_distance = float('inf')
    max_distance = 450  # Increased to detect walls farther away
    
    for contour in contours:
        # Skip tiny contours
        area = cv2.contourArea(contour)
        if area < 100:
            continue
            
        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        
        # Vector from mole to centroid
        vx = centroid_x - cx
        vy = centroid_y - cy
        
        # Distance check
        distance = math.sqrt(vx*vx + vy*vy)
        if distance > max_distance:
            continue
        
        # Track closest wall as fallback
        if distance < closest_distance:
            closest_distance = distance
            closest_contour = contour
        
        # Dot product with RIGHT vector
        # Positive = wall is to the RIGHT of mole
        dot_right = vx * right_x + vy * right_y
        
        # Score: prioritize walls that are CLOSE, then use right-alignment as tie breaker
        if dot_right > 0:  # Positive = right side
            # New Score: Heavy penalty on distance, light bonus for right-alignment
            score = -distance * 5.0 + dot_right * 1.0
            if score > best_score:
                best_score = score
                best_contour = contour
    
    # Return best right-wall, or fallback to closest wall
    if best_contour is not None:
        return best_contour
    else:
        return closest_contour  # Fallback: follow closest wall


def generate_guide_rail(wall_mask, mole_pos=None, heading=0):
    """
    Generate a virtual "guide rail" path using CLOSEST-POINT wall selection.
    
    Instead of selecting walls by centroid (which jumps at corners), we find
    the wall point closest to the mole's right side and generate the rail
    from that wall contour.
    
    Returns: guide_rail image (white line on black background)
    """
    h, w = wall_mask.shape
    
    if mole_pos is None:
        return np.zeros_like(wall_mask)
    
    cx, cy = mole_pos
    
    # Calculate right direction vector
    heading_rad = math.radians(heading)
    right_x = math.cos(heading_rad)  # Perpendicular to forward
    right_y = math.sin(heading_rad)
    
    # Sample point to the right of the mole (where we expect the wall)
    sample_dist = 100  # How far to the right to look
    sample_x = int(cx + right_x * sample_dist)
    sample_y = int(cy + right_y * sample_dist)
    
    # Find all wall contours
    wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(wall_contours) == 0:
        return np.zeros_like(wall_mask)
    
    # Find the wall contour with the closest point to our right-side sample
    best_contour = None
    best_distance = float('inf')
    
    for contour in wall_contours:
        if cv2.contourArea(contour) < 100:
            continue
        
        # Find distance from sample point to this contour
        # pointPolygonTest returns negative distance if outside
        dist = abs(cv2.pointPolygonTest(contour, (sample_x, sample_y), True))
        
        if dist < best_distance:
            best_distance = dist
            best_contour = contour
    
    # If no good match at sample point, try closer to mole
    if best_contour is None or best_distance > 150:
        # Try at shorter distances
        for try_dist in [50, 30, 20]:
            sample_x = int(cx + right_x * try_dist)
            sample_y = int(cy + right_y * try_dist)
            
            for contour in wall_contours:
                if cv2.contourArea(contour) < 100:
                    continue
                dist = abs(cv2.pointPolygonTest(contour, (sample_x, sample_y), True))
                if dist < best_distance:
                    best_distance = dist
                    best_contour = contour
    
    if best_contour is None:
        return np.zeros_like(wall_mask)
    
    # Create mask with the selected wall
    right_wall_mask = np.zeros_like(wall_mask)
    cv2.drawContours(right_wall_mask, [best_contour], -1, 255, cv2.FILLED)
    
    # MORPHOLOGICAL FILLET: Dilate + Erode for smooth offset
    total_dilation = GUIDE_RAIL_OFFSET + CURVE_RADIUS
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (total_dilation*2, total_dilation*2))
    dilated = cv2.dilate(right_wall_mask, dilate_kernel, iterations=1)
    
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CURVE_RADIUS*2, CURVE_RADIUS*2))
    filleted = cv2.erode(dilated, erode_kernel, iterations=1)
    
    # Extract the edge as guide rail
    guide_rail = np.zeros_like(wall_mask)
    path_contours, _ = cv2.findContours(filleted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(guide_rail, path_contours, -1, 255, 9)
    
    return guide_rail


def read_virtual_sensors(guide_rail, mole_pos, heading):
    """
    Sample 5 virtual sensors in reverse-U arrangement ahead of the mole.
    
    Returns: (sensor_data, sensor_positions)
        sensor_data: list of intensities for each sensor
        sensor_positions: list of (x, y) positions for visualization
    """
    if mole_pos is None:
        return ([0] * len(SENSOR_ARRAY), [(0, 0)] * len(SENSOR_ARRAY))
    
    h, w = guide_rail.shape
    cx, cy = mole_pos
    
    # Convert heading to radians (heading: 0=Up, 90=Right)
    heading_rad = math.radians(heading - 90)  # Convert to standard math angle
    
    # Calculate forward direction vector
    fwd_x = math.cos(heading_rad)
    fwd_y = math.sin(heading_rad)
    
    # Calculate lateral (perpendicular) direction vector (90° to the right)
    lat_x = -fwd_y  # Perpendicular
    lat_y = fwd_x
    
    # Sample ROI helper
    half = SENSOR_SIZE // 2
    
    def sample_roi(img, x, y):
        x1 = max(0, x - half)
        x2 = min(w, x + half)
        y1 = max(0, y - half)
        y2 = min(h, y + half)
        if x1 >= x2 or y1 >= y2:
            return 0
        roi = img[y1:y2, x1:x2]
        return np.sum(roi > 127)  # Count white pixels
    
    sensor_data = []
    sensor_positions = []
    
    for sensor in SENSOR_ARRAY:
        # Calculate sensor position
        sx = int(cx + sensor['forward'] * fwd_x + sensor['lateral'] * lat_x)
        sy = int(cy + sensor['forward'] * fwd_y + sensor['lateral'] * lat_y)
        
        intensity = sample_roi(guide_rail, sx, sy)
        sensor_data.append(intensity)
        sensor_positions.append((sx, sy))
    
    return (sensor_data, sensor_positions)


def compute_line_steering(sensor_data, dt=0.05):
    """
    PID control based on 5-sensor weighted error.
    
    Weighted error = Σ(intensity_i * weight_i)
    Positive error = line is LEFT = turn LEFT
    Negative error = line is RIGHT = turn RIGHT
    """
    # Initialize static variables for PID memory
    if not hasattr(compute_line_steering, "integral"):
        compute_line_steering.integral = 0.0
        compute_line_steering.prev_error = 0.0
    
    # Weighted Error Calculation
    error = 0.0
    for i, sensor in enumerate(SENSOR_ARRAY):
        error += sensor_data[i] * sensor['weight']
    
    # PID Terms
    # P-term
    p_term = LINE_FOLLOW_KP * error
    
    # I-term (with anti-windup clamping)
    compute_line_steering.integral += error * dt
    compute_line_steering.integral = max(-500, min(500, compute_line_steering.integral))
    i_term = LINE_FOLLOW_KI * compute_line_steering.integral
    
    # D-term
    derivative = (error - compute_line_steering.prev_error) / dt
    d_term = LINE_FOLLOW_KD * derivative
    
    # Total Steering
    steering = p_term + i_term + d_term
    
    # Update memory
    compute_line_steering.prev_error = error
    
    vl = LINE_FOLLOW_BASE_SPEED - steering
    vr = LINE_FOLLOW_BASE_SPEED + steering
    
    # Clamp velocities
    vl = max(-3.0, min(3.0, vl))
    vr = max(-3.0, min(3.0, vr))
    
    return (vl, vr, error)


def calculate_visual_heading(guide_rail, mole_pos, current_heading):
    """
    Estimate the absolute heading of the path segment using computer vision.
    Returns: corrected_heading (degrees) or None if no clear line
    """
    if mole_pos is None:
        return None
        
    h, w = guide_rail.shape
    cx, cy = mole_pos
    
    # 1. Extract local ROI around mole (lookahead)
    roi_size = 100
    x1 = max(0, cx - roi_size)
    x2 = min(w, cx + roi_size)
    y1 = max(0, cy - roi_size)
    y2 = min(h, cy + roi_size)
    
    roi = guide_rail[y1:y2, x1:x2]
    
    # 2. Find contours in ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get largest contour (the path)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
        
    # 3. Fit line
    try:
        [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Fix DeprecationWarning: Extract scalar from 1D array
        vx, vy = vx.item(), vy.item()
        
        # Convert vector to angle (degrees)
        # In image coords: right=(1,0)=0deg, up=(0,-1)=90deg
        # But our system: Up=0, Right=90
        # Image Math: angle = atan2(vy, vx)  (where vy is down)
        # Our System: heading = atan2(vx, -vy) * 180/pi
        
        # We need the line direction that matches our current heading
        # Calculate two possibilities (forward and backward)
        angle1 = math.degrees(math.atan2(vx, -vy)) % 360
        angle2 = (angle1 + 180) % 360
        
        # Pick the one closest to our current belief
        diff1 = abs((angle1 - current_heading + 180) % 360 - 180)
        diff2 = abs((angle2 - current_heading + 180) % 360 - 180)
        
        measured_heading = angle1 if diff1 < diff2 else angle2
        
        # Only trust if difference is reasonable (e.g. < 45 degrees)
        # Otherwise line fitting might be completely wrong (e.g. at T-junction)
        if min(diff1, diff2) < 45:
            return measured_heading
            
    except:
        pass
        
    return None


# STEP 1: MOLE DETECTION FUNCTION
def detect_mole(image):

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow color
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get largest contour and fit circle
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    
    # Ignore tiny detections (noise)
    if radius < 5:
        return None
    
    return (int(x), int(y))


# STEP 2: DRONE TRACKING FUNCTION
def compute_drone_velocity(mole_pos):

    # Stop if Mole lost
    if mole_pos is None:
        return (0.0, 0.0)  
    
    # Calculate error (how far from calibrated center)
    target_x = IMAGE_CENTER[0] + DRONE_CENTER_OFFSET[0]
    target_y = IMAGE_CENTER[1] + DRONE_CENTER_OFFSET[1]
    error_x = mole_pos[0] - target_x
    error_y = mole_pos[1] - target_y
    
    # PID for X axis
    pid_state['integral_x'] += error_x
    pid_state['integral_x'] = max(-500, min(500, pid_state['integral_x']))  #Clamp Integral
    derivative_x = error_x - pid_state['prev_error_x']
    vx = DRONE_KP * error_x + DRONE_KI * pid_state['integral_x'] + DRONE_KD * derivative_x
    pid_state['prev_error_x'] = error_x
    
    # PID for Y axis
    pid_state['integral_y'] += error_y
    pid_state['integral_y'] = max(-500, min(500, pid_state['integral_y']))
    derivative_y = error_y - pid_state['prev_error_y']
    vy = DRONE_KP * error_y + DRONE_KI * pid_state['integral_y'] + DRONE_KD * derivative_y
    vy = -vy  # Flip Y coz of sign
    pid_state['prev_error_y'] = error_y
    
    # Clamp velocity (max speed)
    max_vel = 6.0
    vx = max(-max_vel, min(max_vel, vx))
    vy = max(-max_vel, min(max_vel, vy))
    
    return (vx, vy)


# STEP 3: WALL DETECTION FUNCTION
def detect_walls(image):
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red wraps around in HSV, so we need two masks
    mask1 = cv2.inRange(hsv, WALL_LOWER_1, WALL_UPPER_1)  # Hue 0-10
    mask2 = cv2.inRange(hsv, WALL_LOWER_2, WALL_UPPER_2)  # Hue 170-180
    wall_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    return wall_mask

################# ADD UTILITY FUNCTIONS HERE #################

def get_camera_image(sim):
    '''
	Purpose:
	---
    Function to get camera image from CoppeliaSim world.
    You are NOT allowed to modify this function

	Input Arguments:
	---
    `sim` : ZeroMQ RemoteAPI object
	
	Returns:
	---
	`img` : A single frame of the Hawk's camera output
    '''
    packed = sim.getBufferSignal('hawk_image')
    if packed is None:
        return None

    resX = sim.getInt32Signal('hawk_res_x')
    resY = sim.getInt32Signal('hawk_res_y')

    img = np.frombuffer(packed, dtype=np.uint8)
    img = img.reshape((resY, resX, 3))
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),0)
    return img


def send_mole_velocity(sim, vl, vr) -> None:
    '''
	Purpose:
	---
    Helper function to send velocity commands to the Mole
    You are NOT allowed to modify this function

	Input Arguments:
	---
    `sim` : ZeroMQ RemoteAPI object
    `vl`  : Target Velocity of the left mole wheel
    `vr`  : Target Velocity of the right mole wheel
	
	Returns:
	---
	None
    '''
    sim.setFloatSignal('mole_left_vel', float(vl))
    sim.setFloatSignal('mole_right_vel', float(vr))


def send_hawk_velocity(sim, vx, vy) -> None:
    '''
	Purpose:
	---
    Helper function to send velocity commands to the Hawk
    You are NOT allowed to modify this function
	
	Input Arguments:
	---
    `sim` : ZeroMQ RemoteAPI object
    `vx`  : Target Velocity of the Hawk in the X direction
    `vy`  : Target Velocity of the Hawk in the Y direction
	
	Returns:
	---
	None
    '''
    sim.setFloatSignal('hawk_vx', float(vx))
    sim.setFloatSignal('hawk_vy', float(vy))


##############################################################

def control_logic(sim):
    """
    EVOLVED Control Logic with State Machine.
    
    States:
    - WALL_FOLLOW: Right-wall following (primary)
    - LOOP_ESCAPE: Escape when stuck in a loop
    
    Features:
    - Path detection using vector cross-product
    - Heading via turn counting + delta position
    - Loop detection via position stagnation
    """
    print("=" * 50)
    print("  MAZE ESCAPE EVOLUTION")
    print("  WALL_FOLLOW | LOOP_ESCAPE")
    print("  Press 'q' to quit")
    print("=" * 50)
    
    # STATE MACHINE INITIALIZATION
    if not hasattr(control_logic, 'state'):
        control_logic.state = 'WALL_FOLLOW'
        control_logic.heading = 0.0
        control_logic.turn_count = 0
        control_logic.accumulated_rotation = 0.0
        control_logic.last_pos = None
        control_logic.last_pos_2 = None  # For delta heading
        control_logic.stuck_counter = 0
        control_logic.escape_counter = 0
        control_logic.current_vl = 0.0
        control_logic.current_vr = 0.0
        control_logic.last_pid_time = None
    
    frame_count = 0
    
    while True:
        # Get camera image
        image = get_camera_image(sim)
        if image is None:
            time.sleep(0.05)
            continue
        
        frame_count += 1
        
        # Perception
        
        # Detect Mole position
        mole_pos = detect_mole(image)
        
        # Drone tracking (PID to keep mole centered)
        hawk_vx, hawk_vy = compute_drone_velocity(mole_pos)
        send_hawk_velocity(sim, hawk_vx, hawk_vy)
        
        # Detect Walls (for fallback)
        wall_mask = detect_walls(image)
        
        # Detect Green Path (primary)
        green_mask = detect_green_path(image)
        
        # Heading Estimation
        
        current_time = sim.getSimulationTime()
        
        # Update heading from wheel velocity (turn counting)
        if heading_state['last_time'] is not None:
            dt = current_time - heading_state['last_time']
            omega = (heading_state['last_vl'] - heading_state['last_vr']) * ANGULAR_RATE / 2.0
            heading_state['heading'] = (heading_state['heading'] + omega * dt) % 360
            
            # Track accumulated rotation for turn counting
            control_logic.accumulated_rotation += omega * dt
            
            # Count 90-degree turns
            if abs(control_logic.accumulated_rotation) >= 90:
                if control_logic.accumulated_rotation > 0:
                    control_logic.turn_count += 1
                else:
                    control_logic.turn_count -= 1
                control_logic.accumulated_rotation = 0.0
        
        heading_state['last_time'] = current_time
        
        # Delta heading fallback (2-frame position tracking)
        if mole_pos is not None:
            if control_logic.last_pos is not None and control_logic.last_pos_2 is not None:
                # Calculate movement direction
                dx = mole_pos[0] - control_logic.last_pos_2[0]
                dy = mole_pos[1] - control_logic.last_pos_2[1]
                
                if abs(dx) > 5 or abs(dy) > 5:  # Significant movement
                    # Calculate heading from delta (0=Up, 90=Right)
                    delta_heading = math.degrees(math.atan2(dx, -dy)) % 360
                    # Could blend with wheel-based heading if needed
            
            # Update position history
            control_logic.last_pos_2 = control_logic.last_pos
            control_logic.last_pos = mole_pos
        
        # Loop Detection (Position Stagnation)
        
        if mole_pos is not None and control_logic.last_pos_2 is not None:
            dx = mole_pos[0] - control_logic.last_pos_2[0]
            dy = mole_pos[1] - control_logic.last_pos_2[1]
            movement = math.sqrt(dx*dx + dy*dy)
            
            if movement < 3:  # Not moving
                control_logic.stuck_counter += 1
            else:
                control_logic.stuck_counter = 0
        
        # FSM State Machine Logic
        
        # Generate guide rail
        guide_rail = generate_guide_rail(wall_mask, mole_pos, heading_state['heading'])
        
        # State Transitions
        prev_state = control_logic.state
        
        if control_logic.state == 'LOOP_ESCAPE':
            # In escape mode, count down
            control_logic.escape_counter -= 1
            if control_logic.escape_counter <= 0:
                control_logic.state = 'WALL_FOLLOW'
                control_logic.stuck_counter = 0
        
        elif control_logic.stuck_counter > 80:  # Stuck for ~2 seconds
            control_logic.state = 'LOOP_ESCAPE'
            control_logic.escape_counter = 40  # Escape for ~1 second
        
        # Control Logic per State
        
        if mole_pos is not None:
            # Update heading with Visual Correction (Sensor Fusion)
            visual_heading = calculate_visual_heading(guide_rail, mole_pos, heading_state['heading'])
            
            if visual_heading is not None and control_logic.state == 'WALL_FOLLOW':
                # Complementary Filter: 98% Odometry, 2% Vision
                # This kills long-term drift while keeping smooth motion
                alpha = 0.98
                
                # Handle angle wrap-around for fusion
                h_odom = heading_state['heading']
                h_vis = visual_heading
                
                # Unwrap visual to be close to odom
                diff = (h_vis - h_odom + 180) % 360 - 180
                h_vis_unwrapped = h_odom + diff
                
                new_heading = alpha * h_odom + (1.0 - alpha) * h_vis_unwrapped
                heading_state['heading'] = new_heading % 360

            sensor_data, sensor_positions = read_virtual_sensors(guide_rail, mole_pos, heading_state['heading'])
            
            # Calculate dt for PID
            if control_logic.last_pid_time is None:
                control_logic.last_pid_time = current_time
            pid_dt = current_time - control_logic.last_pid_time
            if pid_dt <= 0: pid_dt = 0.01
            control_logic.last_pid_time = current_time
            
            if control_logic.state == 'LOOP_ESCAPE':
                # Escape: Reverse and turn right
                vl, vr = -1.5, 1.5
                error = 0
            
            else:
                # Normal line following (WALL_FOLLOW)
                vl, vr, error = compute_line_steering(sensor_data, dt=pid_dt)
        else:
            vl, vr = 0.0, 0.0
            sensor_data = [0] * len(SENSOR_ARRAY)
            sensor_positions = [(0, 0)] * len(SENSOR_ARRAY)
            error = 0
        
        # Velocity Ramping (Smooth Acceleration)
        
        ACCEL_STEP = 0.5
        
        # Smoothly ramp left wheel
        if control_logic.current_vl < vl:
            control_logic.current_vl = min(vl, control_logic.current_vl + ACCEL_STEP)
        elif control_logic.current_vl > vl:
            control_logic.current_vl = max(vl, control_logic.current_vl - ACCEL_STEP)
            
        # Smoothly ramp right wheel
        if control_logic.current_vr < vr:
            control_logic.current_vr = min(vr, control_logic.current_vr + ACCEL_STEP)
        elif control_logic.current_vr > vr:
            control_logic.current_vr = max(vr, control_logic.current_vr - ACCEL_STEP)
            
        # Send velocities
        send_mole_velocity(sim, control_logic.current_vl, control_logic.current_vr)
        
        # Update heading state for next frame
        heading_state['last_vl'] = control_logic.current_vl
        heading_state['last_vr'] = control_logic.current_vr
        
        # Visualization
        
        # Draw on camera image
        if mole_pos is not None:
            cv2.circle(image, mole_pos, 25, (0, 255, 0), 2)
        cv2.drawMarker(image, IMAGE_CENTER, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        
        # State label on camera
        state_colors = {
            'WALL_FOLLOW': (0, 255, 255),
            'LOOP_ESCAPE': (0, 0, 255)
        }
        state_color = state_colors.get(control_logic.state, (255, 255, 255))
        cv2.putText(image, control_logic.state, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Create combined display
        wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        
        # Wall guide rail in YELLOW
        guide_color = cv2.cvtColor(guide_rail, cv2.COLOR_GRAY2BGR)
        guide_color[:, :, 0] = 0  # Remove blue
        wall_display = cv2.addWeighted(wall_display, 1.0, guide_color, 1.0, 0)
        
        # Draw mole position with heading arrow
        if mole_pos is not None:
            # Heading arrow (cyan)
            arrow_length = 40
            angle_rad = math.radians(heading_state['heading'] - 90)
            end_x = int(mole_pos[0] + arrow_length * math.cos(angle_rad))
            end_y = int(mole_pos[1] + arrow_length * math.sin(angle_rad))
            cv2.arrowedLine(wall_display, mole_pos, (end_x, end_y), (255, 255, 0), 3, tipLength=0.3)
            
            # Mole center
            cv2.circle(wall_display, mole_pos, 8, (0, 255, 0), -1)
            
            # Draw all 5 sensor boxes with color gradient (blue left to red right)
            half = SENSOR_SIZE // 2
            sensor_colors = [
                (255, 0, 0),    # FAR_L: Blue
                (255, 128, 0),  # LEFT: Light blue
                (0, 255, 0),    # CENTER: Green
                (0, 128, 255),  # RIGHT: Orange
                (0, 0, 255),    # FAR_R: Red
            ]
            for i, (sx, sy) in enumerate(sensor_positions):
                color = sensor_colors[i] if i < len(sensor_colors) else (255, 255, 255)
                cv2.rectangle(wall_display, 
                             (sx - half, sy - half),
                             (sx + half, sy + half),
                             color, 2)
                # Show intensity value
                cv2.putText(wall_display, str(sensor_data[i]), (sx - 10, sy - half - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Display info
            cv2.putText(wall_display, f"{control_logic.state}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
            sensor_str = " ".join([f"{s}:{sensor_data[i]}" for i, s in enumerate(['LL','L','C','R','RR'])])
            cv2.putText(wall_display, sensor_str, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(wall_display, f"H:{heading_state['heading']:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Camera", image)
        cv2.imshow("Guide Rail", wall_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 10 == 0:  # More frequent logging for tuning
            sensor_str = " ".join([f"{sensor_data[i]:3}" for i in range(len(SENSOR_ARRAY))])
            t = frame_count / 30.0  # Approximate time in seconds
            print(f"[{t:5.1f}s] {control_logic.state:12} [{sensor_str}] err:{error:+6.1f} vl:{vl:+5.2f} vr:{vr:+5.2f}")
        
        time.sleep(0.01)
    
    cv2.destroyAllWindows()
    return None

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE MAIN CODE BELOW #########

if __name__ == "__main__":
	client = RemoteAPIClient()
	sim = client.getObject('sim')	

	try:

		## Start the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.startSimulation()
			if sim.getSimulationState() != sim.simulation_stopped:
				print('\nSimulation started correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be started correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be started !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

		## Runs the control logic written by participants
		try:
			control_logic(sim)

		except Exception:
			print('\n[ERROR] Your control_logic function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually if required.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

		
		## Stop the simulation
		try:
			return_code = sim.stopSimulation()
			time.sleep(0.5)
			if sim.getSimulationState() == sim.simulation_stopped:
				print('\nSimulation stopped correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be stopped correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be stopped !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

	except KeyboardInterrupt:
		## Stop the simulation
		return_code = sim.stopSimulation()
		time.sleep(0.5)
		if sim.getSimulationState() == sim.simulation_stopped:
			print('\nSimulation interrupted by user in CoppeliaSim.')
		else:
			print('\nSimulation could not be interrupted. Stop the simulation manually .')
			sys.exit()