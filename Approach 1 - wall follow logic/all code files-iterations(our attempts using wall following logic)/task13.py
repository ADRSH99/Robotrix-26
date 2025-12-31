'''
*****************************************************************************************
*
*        =================================================
*                    IEEE Robotrix 2025-26
*        =================================================
*
*  This script is intended for implementation of the final hackathon
*  task of IEEE Robotrix 2025-26
*
*****************************************************************************************
'''

# Team Name: ROBODIH
# Team Members: Adarsh Bellamane, Aniruddh Yelluri 
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

# PID Constants
DRONE_KP = 0.007   
DRONE_KI = 0.00012
DRONE_KD = 0.023   

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

# NAVIGATION Parameters (ADDED FOR WALL FOLLOWING)
MOLE_BASE_SPEED = 2.0      # Forward speed - reduced for better control
MOLE_SLOW_SPEED = 1.0      # Speed when wall ahead
MOLE_TURN_SPEED = 2.0      # Wheel speed difference for sharp turns
MOLE_MAX_VEL = 4.0         # Max wheel velocity

# Wall following parameters (NEW)
WALL_FOLLOW_DISTANCE = 50   # Distance to keep from right wall (pixels)
DETECTION_DISTANCE = 60     # How far ahead to check for walls (pixels)
CHECK_REGION_SIZE = 30      # Size of region to check for walls

# Timing parameters (NEW)
INITIAL_STRAIGHT_TIME = 2.0  # Mole goes straight for 2 seconds initially
HAWK_DELAY_TIME = 1.0        # Hawk waits 1 second before tracking

# Turn timing parameters (NEW - for fixed duration turns)
TURN_DURATION = 0.3         # Fixed time to turn (seconds)
STRAIGHT_AFTER_TURN = 4.0    # Time to go straight after turn before checking walls again

# Navigation state (NEW)
nav_state = "INITIAL"
start_time = 0

# Turn state variables (NEW)
turn_start_time = 0
turn_type = "NONE"  # "RIGHT", "LEFT", "U_TURN", "NONE"
in_turn = False
post_turn_straight_start = 0
in_post_turn_straight = False

# NEW: Mole turn tracking and orientation history
mole_orientation = 0  # 0: facing forward, 1: turned right, -1: turned left, 2: U-turn completed
orientation_history = []  # Track history of turns
turn_count = 0
last_turn_time = 0

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

##############################################################

# STEP 1: MOLE DETECTION FUNCTION (EXISTING - UNCHANGED)
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


# STEP 2: DRONE TRACKING FUNCTION (EXISTING - UNCHANGED)
def compute_drone_velocity(mole_pos):

    # Stop if Mole lost
    if mole_pos is None:
        return (0.0, 0.0)  
    
    # Calculate error (how far from center)
    error_x = mole_pos[0] - IMAGE_CENTER[0]
    error_y = mole_pos[1] - IMAGE_CENTER[1]
    
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


# STEP 3: WALL DETECTION FUNCTION (EXISTING - UNCHANGED)
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

# ============================================
# NEW FUNCTIONS FOR WALL FOLLOWING WITH TURN-BASED MOLE ADJUSTMENTS
# ============================================

def check_wall_at_position(wall_mask, x, y, direction="front"):
    """
    Check if there's a wall at a specific position relative to Mole
    Returns True if wall detected, False otherwise
    """
    height, width = wall_mask.shape
    
    # Define check position based on direction
    if direction == "front":
        check_x = x
        check_y = max(0, y - DETECTION_DISTANCE)  # Front is up in image
    elif direction == "right":
        check_x = min(width-1, x + DETECTION_DISTANCE)
        check_y = y
    elif direction == "left":
        check_x = max(0, x - DETECTION_DISTANCE)
        check_y = y
    else:
        return False
    
    # Define region around check position
    start_x = max(0, check_x - CHECK_REGION_SIZE//2)
    end_x = min(width, check_x + CHECK_REGION_SIZE//2)
    start_y = max(0, check_y - CHECK_REGION_SIZE//2)
    end_y = min(height, check_y + CHECK_REGION_SIZE//2)
    
    # Check if region is valid
    if start_x >= end_x or start_y >= end_y:
        return False
    
    # Extract region and count wall pixels
    region = wall_mask[start_y:end_y, start_x:end_x]
    wall_pixels = np.sum(region) / 255  # Count white pixels
    
    # If more than 25% of region has walls, consider it a wall
    region_area = (end_x - start_x) * (end_y - start_y)
    return wall_pixels > (0.25 * region_area)

def update_mole_orientation(turn_type, current_time):
    """
    Update the mole's orientation based on turn type
    """
    global mole_orientation, orientation_history, turn_count, last_turn_time
    
    time_since_last_turn = current_time - last_turn_time
    
    # Only update if enough time has passed since last turn (avoid double counting)
    if time_since_last_turn < 0.5:  # 500ms debounce
        return
    
    if turn_type == "RIGHT":
        mole_orientation = 1  # Turned right
        orientation_history.append("RIGHT")
        turn_count += 1
        last_turn_time = current_time
        print(f"[ORIENTATION] Mole turned RIGHT. Orientation: {mole_orientation}")
        
    elif turn_type == "LEFT":
        mole_orientation = -1  # Turned left
        orientation_history.append("LEFT")
        turn_count += 1
        last_turn_time = current_time
        print(f"[ORIENTATION] Mole turned LEFT. Orientation: {mole_orientation}")
        
    elif turn_type == "U_TURN":
        mole_orientation = 2  # Completed U-turn
        orientation_history.append("U_TURN")
        turn_count += 1
        last_turn_time = current_time
        print(f"[ORIENTATION] Mole completed U-TURN. Orientation: {mole_orientation}")
        
    elif turn_type == "STRAIGHT":
        # Reset to forward after straight movement
        mole_orientation = 0  # Facing forward
        print(f"[ORIENTATION] Mole moving STRAIGHT. Orientation: {mole_orientation}")

def adjust_mole_velocity_based_on_orientation(vl, vr, mole_orientation):
    """
    Adjust mole velocities based on its current orientation
    """
    adjusted_vl = vl
    adjusted_vr = vr
    
    # Apply adjustments based on orientation
    if mole_orientation == 1:  # Turned RIGHT
        # After right turn, need to compensate slightly to the left
        adjustment_factor = 1.15
        adjusted_vl *= adjustment_factor
        adjusted_vr *= (2.0 - adjustment_factor)  # Opposite adjustment
        print(f"[ADJUST] After RIGHT turn: vl={adjusted_vl:.2f}, vr={adjusted_vr:.2f}")
        
    elif mole_orientation == -1:  # Turned LEFT
        # After left turn, need to compensate slightly to the right
        adjustment_factor = 0.85
        adjusted_vl *= adjustment_factor
        adjusted_vr *= (2.0 - adjustment_factor)  # Opposite adjustment
        print(f"[ADJUST] After LEFT turn: vl={adjusted_vl:.2f}, vr={adjusted_vr:.2f}")
        
    elif mole_orientation == 2:  # Completed U-TURN
        # After U-turn, significant adjustment needed
        adjustment_factor = 1.3
        adjusted_vl *= adjustment_factor
        adjusted_vr *= (2.0 - adjustment_factor)  # Opposite adjustment
        print(f"[ADJUST] After U-TURN: vl={adjusted_vl:.2f}, vr={adjusted_vr:.2f}")
    
    # Clamp velocities to safe limits
    adjusted_vl = max(-MOLE_MAX_VEL, min(MOLE_MAX_VEL, adjusted_vl))
    adjusted_vr = max(-MOLE_MAX_VEL, min(MOLE_MAX_VEL, adjusted_vr))
    
    return adjusted_vl, adjusted_vr

def compute_mole_velocity(mole_pos, wall_mask, current_time):
    """
    Calculate Mole wheel velocities using right-hand wall following rule
    with FIXED DURATION TURNS and orientation-based adjustments
    Returns (vl, vr, state_description, orientation)
    """
    global nav_state, start_time, in_turn, turn_start_time, turn_type
    global in_post_turn_straight, post_turn_straight_start
    global mole_orientation
    
    elapsed = current_time - start_time
    
    # Initial straight movement
    if elapsed < INITIAL_STRAIGHT_TIME:
        mole_orientation = 0  # Straight
        return MOLE_BASE_SPEED, MOLE_BASE_SPEED, "INITIAL_STRAIGHT", 0
    
    # If Mole not detected, stop
    if mole_pos is None:
        mole_orientation = 0
        return 0.0, 0.0, "MOLENOT_FOUND", 0
    
    # Update navigation state
    if nav_state == "INITIAL" and elapsed >= INITIAL_STRAIGHT_TIME:
        nav_state = "WALL_FOLLOWING"
        print(f"[NAV] Switching to WALL_FOLLOWING at {elapsed:.1f}s")
    
    # ====================================================
    # FIXED DURATION TURN LOGIC
    # ====================================================
    
    # 1. If currently executing a timed turn
    if in_turn:
        turn_elapsed = current_time - turn_start_time
        
        if turn_elapsed < TURN_DURATION:
            # Continue turning for the fixed duration
            if turn_type == "RIGHT":
                vl = MOLE_TURN_SPEED
                vr = -MOLE_TURN_SPEED * 0.8
                state = f"TURN_RIGHT ({turn_elapsed:.1f}/{TURN_DURATION}s)"
            elif turn_type == "LEFT":
                vl = -MOLE_TURN_SPEED * 0.8
                vr = MOLE_TURN_SPEED
                state = f"TURN_LEFT ({turn_elapsed:.1f}/{TURN_DURATION}s)"
            elif turn_type == "U_TURN":
                vl = -MOLE_TURN_SPEED * 0.7
                vr = MOLE_TURN_SPEED * 0.7
                state = f"U_TURN ({turn_elapsed:.1f}/{TURN_DURATION}s)"
            else:
                # Default to straight if turn_type is invalid
                vl = MOLE_BASE_SPEED
                vr = MOLE_BASE_SPEED
                state = "STRAIGHT"
            
            return vl, vr, state, mole_orientation
        else:
            # Turn duration completed, start post-turn straight phase
            in_turn = False
            in_post_turn_straight = True
            post_turn_straight_start = current_time
            
            # Update orientation based on completed turn
            update_mole_orientation(turn_type, current_time)
            
            print(f"[TURN] {turn_type} turn completed after {TURN_DURATION}s")
            vl = MOLE_BASE_SPEED
            vr = MOLE_BASE_SPEED
            
            # Apply orientation-based adjustments to straight movement
            vl, vr = adjust_mole_velocity_based_on_orientation(vl, vr, mole_orientation)
            
            state = "POST_TURN_START"
            return vl, vr, state, mole_orientation
    
    # 2. If in post-turn straight phase
    if in_post_turn_straight:
        straight_elapsed = current_time - post_turn_straight_start
        
        if straight_elapsed < STRAIGHT_AFTER_TURN:
            # Go straight for fixed time after turn
            vl = MOLE_BASE_SPEED
            vr = MOLE_BASE_SPEED
            
            # Apply orientation-based adjustments
            vl, vr = adjust_mole_velocity_based_on_orientation(vl, vr, mole_orientation)
            
            state = f"POST_TURN_STRAIGHT ({straight_elapsed:.1f}/{STRAIGHT_AFTER_TURN}s)"
            return vl, vr, state, mole_orientation
        else:
            # Post-turn straight completed, resume normal wall following
            in_post_turn_straight = False
            update_mole_orientation("STRAIGHT", current_time)  # Reset to straight orientation
            print(f"[TURN] Post-turn straight completed, resuming wall following")
    
    # 3. Normal wall following decision making (only if not in turn or post-turn phase)
    # Check walls in different directions
    front_wall = check_wall_at_position(wall_mask, mole_pos[0], mole_pos[1], "front")
    right_wall = check_wall_at_position(wall_mask, mole_pos[0], mole_pos[1], "right")
    left_wall = check_wall_at_position(wall_mask, mole_pos[0], mole_pos[1], "left")

    # For debugging 
    print(f"[WALLS] Front: {front_wall}, Right: {right_wall}, Left: {left_wall}")
    
    # RIGHT-HAND RULE DECISION MAKING
    if front_wall:
        # Wall in front - need to turn
        if not right_wall:
            # No wall on right, start RIGHT turn
            in_turn = True
            turn_type = "RIGHT"
            turn_start_time = current_time
            vl = MOLE_TURN_SPEED
            vr = -MOLE_TURN_SPEED * 0.8
            print(f"[TURN] Starting RIGHT turn for {TURN_DURATION}s")
            return vl, vr, "START_TURN_RIGHT", mole_orientation
        elif not left_wall:
            # Wall on right but not left, start LEFT turn
            in_turn = True
            turn_type = "LEFT"
            turn_start_time = current_time
            vl = -MOLE_TURN_SPEED * 0.8
            vr = MOLE_TURN_SPEED
            print(f"[TURN] Starting LEFT turn for {TURN_DURATION}s")
            return vl, vr, "START_TURN_LEFT", mole_orientation
        else:
            # Dead end - start U-turn
            in_turn = True
            turn_type = "U_TURN"
            turn_start_time = current_time
            vl = -MOLE_TURN_SPEED * 0.7
            vr = MOLE_TURN_SPEED * 0.7
            print(f"[TURN] Starting U-TURN for {TURN_DURATION}s")
            return vl, vr, "START_U_TURN", mole_orientation
    else:
        # No wall in front
        if not right_wall:
            # No wall on right, curve slightly right to find wall
            vl = MOLE_BASE_SPEED * 1.1
            vr = MOLE_BASE_SPEED * 0.9
            
            # Apply orientation-based adjustments
            vl, vr = adjust_mole_velocity_based_on_orientation(vl, vr, mole_orientation)
            
            return vl, vr, "CURVE_RIGHT", mole_orientation
        else:
            # Wall on right, follow it straight
            vl = MOLE_BASE_SPEED
            vr = MOLE_BASE_SPEED
            
            # Apply orientation-based adjustments
            vl, vr = adjust_mole_velocity_based_on_orientation(vl, vr, mole_orientation)
            
            return vl, vr, "FOLLOW_WALL", mole_orientation

def update_hawk_velocity_with_delay(mole_pos, current_time):
    """
    Calculate Hawk velocity with initial delay
    """
    global start_time
    
    elapsed = current_time - start_time
    
    # Hawk waits before starting to track
    if elapsed < HAWK_DELAY_TIME:
        return 0.0, 0.0, "WAITING"
    
    # Use existing tracking function
    hawk_vx, hawk_vy = compute_drone_velocity(mole_pos)
    
    # Apply additional smoothing for stability
    max_hawk_speed = 4.0
    hawk_vx = max(-max_hawk_speed, min(max_hawk_speed, hawk_vx))
    hawk_vy = max(-max_hawk_speed, min(max_hawk_speed, hawk_vy))
    
    status = "TRACKING" if mole_pos is not None else "MOLENOT_FOUND"
    return hawk_vx, hawk_vy, status

def draw_navigation_info(image, mole_pos, wall_mask, mole_vl, mole_vr, 
                         hawk_vx, hawk_vy, mole_state, hawk_state, 
                         elapsed_time, mole_orientation):
    """
    Draw navigation information on image for debugging
    """
    global in_turn, turn_type, turn_start_time, in_post_turn_straight, post_turn_straight_start
    global orientation_history, turn_count
    
    display = image.copy()
    height, width = display.shape[:2]
    
    # Draw Mole if detected
    if mole_pos is not None:
        # Draw Mole position with orientation color coding
        color = (0, 255, 0)  # Green for straight/forward
        if mole_orientation == 1:  # Turned right
            color = (255, 165, 0)  # Orange
        elif mole_orientation == -1:  # Turned left
            color = (0, 255, 255)  # Yellow
        elif mole_orientation == 2:  # U-turn completed
            color = (255, 0, 255)  # Magenta
            
        cv2.circle(display, mole_pos, 25, color, 2)
        
        # Draw detection zones
        # Front detection zone
        front_y = max(0, mole_pos[1] - DETECTION_DISTANCE)
        cv2.rectangle(display, 
                     (mole_pos[0] - CHECK_REGION_SIZE//2, front_y - CHECK_REGION_SIZE//2),
                     (mole_pos[0] + CHECK_REGION_SIZE//2, front_y + CHECK_REGION_SIZE//2),
                     (0, 255, 255), 1)  # Cyan for front
        
        # Right detection zone
        right_x = min(width-1, mole_pos[0] + DETECTION_DISTANCE)
        cv2.rectangle(display,
                     (right_x - CHECK_REGION_SIZE//2, mole_pos[1] - CHECK_REGION_SIZE//2),
                     (right_x + CHECK_REGION_SIZE//2, mole_pos[1] + CHECK_REGION_SIZE//2),
                     (255, 255, 0), 1)  # Light blue for right
        
        # Left detection zone
        left_x = max(0, mole_pos[0] - DETECTION_DISTANCE)
        cv2.rectangle(display,
                     (left_x - CHECK_REGION_SIZE//2, mole_pos[1] - CHECK_REGION_SIZE//2),
                     (left_x + CHECK_REGION_SIZE//2, mole_pos[1] + CHECK_REGION_SIZE//2),
                     (255, 0, 255), 1)  # Magenta for left
    
    # Draw image center
    cv2.drawMarker(display, IMAGE_CENTER, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    
    # Add navigation status text
    y_offset = 30
    cv2.putText(display, f"State: {nav_state}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(display, f"Time: {elapsed_time:.1f}s", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    # Mole orientation indicator
    orientation_text = "FORWARD"
    if mole_orientation == 1:
        orientation_text = "RIGHT_TURNED"
    elif mole_orientation == -1:
        orientation_text = "LEFT_TURNED"
    elif mole_orientation == 2:
        orientation_text = "U_TURNED"
    
    color = (0, 255, 0)
    if mole_orientation == 1:
        color = (255, 165, 0)
    elif mole_orientation == -1:
        color = (0, 255, 255)
    elif mole_orientation == 2:
        color = (255, 0, 255)
        
    cv2.putText(display, f"Mole Orientation: {orientation_text}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y_offset += 30
    cv2.putText(display, f"Turn Count: {turn_count}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(display, f"Mole State: {mole_state}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 30
    cv2.putText(display, f"Hawk State: {hawk_state}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y_offset += 30
    cv2.putText(display, f"Mole Vel: ({mole_vl:.1f}, {mole_vr:.1f})", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(display, f"Hawk Vel: ({hawk_vx:.2f}, {hawk_vy:.2f})", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add turn history
    y_offset += 25
    if len(orientation_history) > 0:
        recent_turns = orientation_history[-3:]  # Show last 3 turns
        history_text = f"Recent Turns: {recent_turns}"
        cv2.putText(display, history_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Add turn status if applicable
    y_offset += 25
    if in_turn:
        current_time_val = time.time()
        turn_elapsed = current_time_val - turn_start_time
        remaining = TURN_DURATION - turn_elapsed
        cv2.putText(display, f"TURNING {turn_type}: {remaining:.1f}s left", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Orange
    elif in_post_turn_straight:
        current_time_val = time.time()
        straight_elapsed = current_time_val - post_turn_straight_start
        remaining = STRAIGHT_AFTER_TURN - straight_elapsed
        cv2.putText(display, f"POST-TURN STRAIGHT: {remaining:.1f}s left", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow
    
    # Create composite with wall mask
    wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
    composite = np.hstack([display, wall_display])
    
    return composite

# ============================================
# UTILITY FUNCTIONS (EXISTING - UNCHANGED)
# ============================================

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

# ============================================
# MAIN CONTROL LOGIC (MODIFIED FOR WALL FOLLOWING)
# ============================================

def control_logic(sim):
    """
    Main control logic with wall following navigation and FIXED DURATION TURNS
    """
    global nav_state, start_time, in_turn, turn_type, turn_start_time
    global in_post_turn_straight, post_turn_straight_start
    global mole_orientation, orientation_history, turn_count, last_turn_time
    
    # Reset all states
    nav_state = "INITIAL"
    start_time = time.time()
    in_turn = False
    turn_type = "NONE"
    turn_start_time = 0
    in_post_turn_straight = False
    post_turn_straight_start = 0
    mole_orientation = 0
    orientation_history = []
    turn_count = 0
    last_turn_time = 0
    
    print("=" * 60)
    print("WALL FOLLOWING WITH ORIENTATION-BASED MOLE ADJUSTMENTS")
    print("Strategy: Right-hand rule with timed turns")
    print("Mole adjusts its own velocity based on turn history")
    print("=" * 60)
    print(f"\nTIMING PARAMETERS:")
    print(f"1. Initial straight: {INITIAL_STRAIGHT_TIME}s")
    print(f"2. Turn duration: {TURN_DURATION}s")
    print(f"3. Post-turn straight: {STRAIGHT_AFTER_TURN}s")
    print(f"4. Hawk delay: {HAWK_DELAY_TIME}s")
    print("\nORIENTATION STATES:")
    print("0: Forward, 1: Turned Right, -1: Turned Left, 2: U-Turned")
    print("\nStarting navigation...")
    
    frame_count = 0
    
    try:
        while True:
            # Get camera image
            image = get_camera_image(sim)
            if image is None:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Step 1: Detect Mole (using existing function)
            mole_pos = detect_mole(image)
            
            # Step 2: Detect walls (using existing function)
            wall_mask = detect_walls(image)
            
            # Step 3: Calculate Mole velocity for wall following WITH orientation-based adjustments
            mole_vl, mole_vr, mole_state, mole_orientation = compute_mole_velocity(mole_pos, wall_mask, current_time)
            
            # Step 4: Calculate Hawk velocity with delay
            hawk_vx, hawk_vy, hawk_state = update_hawk_velocity_with_delay(mole_pos, current_time)
            
            # Step 5: Send velocity commands
            send_mole_velocity(sim, mole_vl, mole_vr)
            send_hawk_velocity(sim, hawk_vx, hawk_vy)
            
            # Step 6: Visualize (every 3 frames for performance)
            if frame_count % 3 == 0:
                debug_image = draw_navigation_info(
                    image, mole_pos, wall_mask, 
                    mole_vl, mole_vr, hawk_vx, hawk_vy,
                    mole_state, hawk_state, elapsed_time, mole_orientation
                )
                
                cv2.imshow('Wall Following with Orientation-based Mole Adjustments', debug_image)
            
            # Step 7: Print status periodically
            if frame_count % 30 == 0:
                print(f"[{elapsed_time:.1f}s] State: {nav_state}, Mole: {mole_state}")
                print(f"       Mole Orientation: {mole_orientation}")
                print(f"       Turn Count: {turn_count}")
                print(f"       Mole Vel: ({mole_vl:.1f}, {mole_vr:.1f})")
                print(f"       Hawk Vel: ({hawk_vx:.2f}, {hawk_vy:.2f})")
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("\nNavigation stopped by user")
                break
            
            # Small delay for stability
            time.sleep(0.03)
            
    except Exception as e:
        print(f"\n[ERROR] Exception in control logic: {e}")
        traceback.print_exc()
    
    finally:
        # Ensure we stop both bots when exiting
        send_mole_velocity(sim, 0, 0)
        send_hawk_velocity(sim, 0, 0)


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