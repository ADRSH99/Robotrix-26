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

# STEP 2: DRONE TRACKING
IMAGE_CENTER = (256, 256)  # Center of 512x512 image

# MUCH SMALLER PID Constants to prevent oscillations
DRONE_KP = 0.0007   # Reduced by 10x
DRONE_KI = 0.00001  # Reduced
DRONE_KD = 0.001    # Reduced

# PID state
pid_state = {
    'integral_x': 0.0,
    'integral_y': 0.0,
    'prev_error_x': 0.0,
    'prev_error_y': 0.0
}

# STEP 3: WALL DETECTION - HSV Thresholds
WALL_LOWER_1 = np.array([0, 100, 100])
WALL_UPPER_1 = np.array([10, 255, 255])
WALL_LOWER_2 = np.array([170, 100, 100])
WALL_UPPER_2 = np.array([180, 255, 255])

# NAVIGATION Parameters (lowered for stability)
MOLE_BASE_SPEED = 1.5      # Forward speed
MOLE_TURN_SPEED = 1.0      # Wheel speed difference for sharp turns
MOLE_MAX_VEL = 3.0         # Max wheel velocity

# Timing parameters
INITIAL_MOLE_TIME = 2.0    # Mole goes straight for 2 seconds
INITIAL_HAWK_TIME = 3.0    # Hawk waits 3 seconds before tracking

# State machine
current_state = "INITIAL"
start_time = time.time()

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

##############################################################

# STEP 1: MOLE DETECTION FUNCTION (from backup)
def detect_mole(image):
    """Detect the yellow Mole with stable detection from backup"""
    
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
        return None, None
    
    # Get largest contour and fit circle
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    
    # Ignore tiny detections (noise)
    if radius < 5:
        return None, None
    
    x_int, y_int = int(x), int(y)
    
    # Return both center and bounding box
    bbox = (int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))
    
    return (x_int, y_int), bbox


# STEP 2: STABILIZED DRONE TRACKING FUNCTION
def compute_drone_velocity(mole_pos, current_time):
    """Calculate Hawk velocity with stabilization to prevent circling"""
    
    global start_time, current_state
    
    # Stop if Mole lost or in initial phase
    elapsed = current_time - start_time
    if mole_pos is None or elapsed < INITIAL_HAWK_TIME:
        # Reset PID to prevent windup
        pid_state['integral_x'] = 0.0
        pid_state['integral_y'] = 0.0
        pid_state['prev_error_x'] = 0.0
        pid_state['prev_error_y'] = 0.0
        return (0.0, 0.0), "WAITING"
    
    # Calculate error (how far from center)
    error_x = mole_pos[0] - IMAGE_CENTER[0]
    error_y = mole_pos[1] - IMAGE_CENTER[1]
    
    # Deadzone - don't move if error is small
    deadzone = 60  # pixels
    if abs(error_x) < deadzone and abs(error_y) < deadzone:
        # Small error, just hover
        return (0.0, 0.0), "HOVERING"
    
    # Apply gentle low-pass filter to errors
    error_x = error_x * 0.7 + pid_state['prev_error_x'] * 0.3
    error_y = error_y * 0.7 + pid_state['prev_error_y'] * 0.3
    
    # PID for X axis (with anti-windup)
    pid_state['integral_x'] += error_x * 0.1  # Reduced integral effect
    # Clamp integral to prevent windup
    pid_state['integral_x'] = max(-100, min(100, pid_state['integral_x']))
    
    derivative_x = error_x - pid_state['prev_error_x']
    
    # Calculate PID output with very small gains
    vx = (DRONE_KP * error_x + 
          DRONE_KI * pid_state['integral_x'] + 
          DRONE_KD * derivative_x)
    
    pid_state['prev_error_x'] = error_x
    
    # PID for Y axis (with anti-windup)
    pid_state['integral_y'] += error_y * 0.1  # Reduced integral effect
    pid_state['integral_y'] = max(-100, min(100, pid_state['integral_y']))
    
    derivative_y = error_y - pid_state['prev_error_y']
    vy = (DRONE_KP * error_y + 
          DRONE_KI * pid_state['integral_y'] + 
          DRONE_KD * derivative_y)
    vy = -vy  # Flip Y because of coordinate system
    pid_state['prev_error_y'] = error_y
    
    # CLAMP VELOCITY VERY AGGRESSIVELY (prevent oscillations)
    max_vel = 0.3  # VERY slow maximum speed
    vx = max(-max_vel, min(max_vel, vx))
    vy = max(-max_vel, min(max_vel, vy))
    
    # Additional damping: if velocity too high, reduce it
    speed = math.sqrt(vx*vx + vy*vy)
    if speed > max_vel * 0.8:
        vx *= 0.5
        vy *= 0.5
    
    return (vx, vy), "TRACKING"


# STEP 3: WALL DETECTION FUNCTION (from backup)
def detect_walls(image):
    """Detect red walls with stable detection from backup"""
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red wraps around in HSV, so we need two masks
    mask1 = cv2.inRange(hsv, WALL_LOWER_1, WALL_UPPER_1)  # Hue 0-10
    mask2 = cv2.inRange(hsv, WALL_LOWER_2, WALL_UPPER_2)  # Hue 170-180
    wall_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    
    # Additional dilation to connect nearby walls
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    
    return wall_mask


# STEP 4: WALL PROXIMITY CHECKING
def check_wall_proximity(wall_mask, mole_pos, direction="front", check_distance=40):
    """Check for walls in specific directions"""
    if mole_pos is None:
        return False
    
    height, width = wall_mask.shape
    x, y = mole_pos
    
    if direction == "front":
        # Check in front (upwards in image)
        check_y = max(0, y - check_distance)
        start_y = max(0, check_y - 15)
        end_y = max(0, check_y + 15)
        start_x = max(0, x - 25)
        end_x = min(width, x + 25)
        
    elif direction == "right":
        # Check to the right
        check_x = min(width-1, x + check_distance)
        start_x = max(0, check_x - 15)
        end_x = min(width, check_x + 15)
        start_y = max(0, y - 25)
        end_y = min(height, y + 25)
        
    elif direction == "left":
        # Check to the left
        check_x = max(0, x - check_distance)
        start_x = max(0, check_x - 15)
        end_x = min(width, check_x + 15)
        start_y = max(0, y - 25)
        end_y = min(height, y + 25)
    else:
        return False
    
    # Ensure valid region
    if start_x >= end_x or start_y >= end_y:
        return False
    
    # Extract region and count wall pixels
    region = wall_mask[start_y:end_y, start_x:end_x]
    wall_pixels = np.sum(region) / 255
    
    # If more than 20% of region has walls, consider it a wall
    region_area = (end_x - start_x) * (end_y - start_y)
    return wall_pixels > (0.2 * region_area)


# STEP 5: MOLE NAVIGATION WITH RIGHT-HAND RULE
def compute_mole_velocity(mole_pos, wall_mask, current_time):
    """Calculate Mole wheel velocities for maze navigation"""
    global start_time, current_state
    
    elapsed = current_time - start_time
    
    # Initial straight movement
    if elapsed < INITIAL_MOLE_TIME:
        return MOLE_BASE_SPEED, MOLE_BASE_SPEED, "INITIAL_STRAIGHT", False, False
    
    # If Mole lost, stop
    if mole_pos is None:
        return 0.0, 0.0, "LOST", False, False
    
    # Check walls in different directions
    front_wall = check_wall_proximity(wall_mask, mole_pos, "front", 50)
    right_wall = check_wall_proximity(wall_mask, mole_pos, "right", 40)
    left_wall = check_wall_proximity(wall_mask, mole_pos, "left", 40)
    
    # SIMPLE RIGHT-HAND RULE
    if front_wall:
        # Wall in front - need to turn
        if not right_wall:
            # No wall on right, turn RIGHT
            vl = MOLE_TURN_SPEED
            vr = -MOLE_TURN_SPEED * 0.8
            return vl, vr, "TURN_RIGHT", front_wall, right_wall
        elif not left_wall:
            # Wall on right, no wall on left, turn LEFT
            vl = -MOLE_TURN_SPEED * 0.8
            vr = MOLE_TURN_SPEED
            return vl, vr, "TURN_LEFT", front_wall, right_wall
        else:
            # Dead end - turn around
            vl = -MOLE_TURN_SPEED * 0.7
            vr = MOLE_TURN_SPEED * 0.7
            return vl, vr, "TURN_AROUND", front_wall, right_wall
    else:
        # No wall in front
        if not right_wall:
            # No wall on right, curve slightly right
            vl = MOLE_BASE_SPEED * 1.1
            vr = MOLE_BASE_SPEED * 0.9
            return vl, vr, "CURVE_RIGHT", front_wall, right_wall
        else:
            # Wall on right, follow it straight
            vl = MOLE_BASE_SPEED
            vr = MOLE_BASE_SPEED
            return vl, vr, "FOLLOW_WALL", front_wall, right_wall


# STEP 6: STATE MACHINE MANAGEMENT
def update_state_machine(mole_pos, current_time):
    """Update system state based on time and Mole detection"""
    global current_state, start_time
    
    elapsed = current_time - start_time
    
    if current_state == "INITIAL":
        if elapsed > INITIAL_MOLE_TIME:
            current_state = "TRACKING"
            print(f"[STATE] TRACKING started at {elapsed:.1f}s")
    
    elif current_state == "TRACKING":
        if mole_pos is None:
            # Mole lost in tracking phase
            print("[STATE] Mole lost during tracking")
        elif elapsed > INITIAL_HAWK_TIME + 2.0:  # After tracking for 2 seconds
            current_state = "NAVIGATING"
            print(f"[STATE] NAVIGATING started at {elapsed:.1f}s")
    
    elif current_state == "NAVIGATING":
        # Main navigation state
        pass
    
    return current_state


# STEP 7: VISUALIZATION FUNCTION
def draw_debug_info(image, mole_pos, mole_bbox, wall_mask, 
                    mole_vl, mole_vr, hawk_vx, hawk_vy, 
                    mole_status, hawk_status, state):
    """Draw debugging information on image"""
    display = image.copy()
    height, width = display.shape[:2]
    
    # Draw Mole if detected
    if mole_bbox is not None:
        x, y, w, h = mole_bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if mole_pos is not None:
        cv2.circle(display, mole_pos, 5, (0, 0, 255), -1)
        
        # Draw detection zones
        # Front zone
        front_y = max(0, mole_pos[1] - 50)
        cv2.rectangle(display, 
                     (mole_pos[0] - 25, front_y - 15),
                     (mole_pos[0] + 25, front_y + 15),
                     (0, 255, 255), 1)
        
        # Right zone
        right_x = min(width-1, mole_pos[0] + 40)
        cv2.rectangle(display,
                     (right_x - 15, mole_pos[1] - 25),
                     (right_x + 15, mole_pos[1] + 25),
                     (255, 255, 0), 1)
    
    # Add status text
    elapsed = time.time() - start_time
    
    cv2.putText(display, f"State: {state}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Time: {elapsed:.1f}s", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Mole: {mole_status}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Hawk: {hawk_status}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"Mole Vel: ({mole_vl:.1f}, {mole_vr:.1f})", (10, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, f"Hawk Vel: ({hawk_vx:.3f}, {hawk_vy:.3f})", (10, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Create composite with wall mask
    wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
    composite = np.hstack([display, wall_display])
    
    return composite


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


def control_logic(sim):
    """
    Purpose:
    ---
    Main control logic combining stable detection with smooth navigation
    """
    global start_time, current_state
    
    print("=" * 60)
    print("STABLE MAZE NAVIGATION SYSTEM")
    print("Using backup detection with stabilized control")
    print("=" * 60)
    
    print("\nPHASES:")
    print(f"1. INITIAL (0-{INITIAL_MOLE_TIME}s): Mole goes straight")
    print(f"2. TRACKING ({INITIAL_MOLE_TIME}-{INITIAL_HAWK_TIME}s): Hawk starts tracking")
    print(f"3. NAVIGATING ({INITIAL_HAWK_TIME}s+): Full wall following")
    print("\nStarting in 1 second...")
    time.sleep(1.0)
    
    frame_count = 0
    
    try:
        while True:
            # Get camera image
            image = get_camera_image(sim)
            if image is None:
                time.sleep(0.02)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Step 1: Detect Mole (using stable backup detection)
            mole_pos, mole_bbox = detect_mole(image)
            
            # Step 2: Detect walls
            wall_mask = detect_walls(image)
            
            # Step 3: Update state machine
            state = update_state_machine(mole_pos, current_time)
            
            # Step 4: Calculate Hawk velocity (with stabilization)
            hawk_vel, hawk_status = compute_drone_velocity(mole_pos, current_time)
            hawk_vx, hawk_vy = hawk_vel
            
            # Step 5: Calculate Mole velocity (right-hand rule)
            mole_vl, mole_vr, mole_status, front_wall, right_wall = compute_mole_velocity(
                mole_pos, wall_mask, current_time
            )
            
            # Step 6: Send velocity commands
            send_mole_velocity(sim, mole_vl, mole_vr)
            
            # Additional safety: limit Hawk speed even more
            max_hawk_speed = 0.25
            if abs(hawk_vx) > max_hawk_speed or abs(hawk_vy) > max_hawk_speed:
                scale = max_hawk_speed / max(abs(hawk_vx), abs(hawk_vy))
                hawk_vx *= scale
                hawk_vy *= scale
            
            send_hawk_velocity(sim, hawk_vx, hawk_vy)
            
            # Step 7: Visualize
            if frame_count % 5 == 0:  # Update display every 5 frames
                debug_image = draw_debug_info(
                    image, mole_pos, mole_bbox, wall_mask,
                    mole_vl, mole_vr, hawk_vx, hawk_vy,
                    mole_status, hawk_status, state
                )
                cv2.imshow('Stable Maze Navigation', debug_image)
            
            # Check for exit
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("\nSimulation stopped by user")
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
        cv2.destroyAllWindows()
        print("\nSystem stopped safely")
    
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