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
# (Avoid global state as much as possible)
##############################################################

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

def detect_mole(image):
    """
    Detect the yellow marker on the Mole using color thresholding.
    Returns the bounding box and center coordinates.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color (for the Mole's marker)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the Mole)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Filter noise
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            return (x, y, w, h), (center_x, center_y)
    
    return None, None


def detect_walls(image):
    """
    Detect red walls in the maze.
    Returns wall mask.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (walls are red)
    # Red in HSV has two ranges (wraps around 0)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    wall_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)
    
    return wall_mask


def check_wall_at_position(wall_mask, x, y, region_size=30):
    """
    Check if there's a wall at the specified position.
    Returns True if wall detected, False otherwise.
    """
    height, width = wall_mask.shape[:2]
    
    # Define region to check
    start_x = max(0, x - region_size//2)
    end_x = min(width, x + region_size//2)
    start_y = max(0, y - region_size//2)
    end_y = min(height, y + region_size//2)
    
    # Check if region is valid
    if start_x >= end_x or start_y >= end_y:
        return False
    
    # Extract region and check for walls
    region = wall_mask[start_y:end_y, start_x:end_x]
    wall_pixels = np.sum(region) / 255  # Count white pixels
    
    # Threshold: if more than 20% of region has walls, return True
    region_area = (end_x - start_x) * (end_y - start_y)
    return wall_pixels > (0.2 * region_area)


def calculate_hawk_velocity(mole_center, image_shape):
    """
    Calculate Hawk velocity to keep Mole in camera frame.
    Returns vx, vy velocities.
    """
    if mole_center is None:
        return 0, 0
    
    height, width = image_shape[:2]
    target_x, target_y = width // 2, height // 2
    mole_x, mole_y = mole_center
    
    # Calculate error from center
    error_x = mole_x - target_x
    error_y = mole_y - target_y
    
    # Deadzone - don't move if Mole is close to center
    deadzone = 50
    
    vx, vy = 0, 0
    
    if abs(error_x) > deadzone:
        vx = -0.002 * error_x  # Negative for opposite movement
    
    if abs(error_y) > deadzone:
        vy = -0.002 * error_y
    
    # Limit speed
    vx = max(-0.5, min(0.5, vx))
    vy = max(-0.5, min(0.5, vy))
    
    return 0.1*vx, 0.1*vy


def calculate_mole_velocity(mole_center, wall_mask, image_shape):
    """
    Calculate Mole velocity using right-wall following algorithm.
    Returns left and right wheel velocities.
    """
    if mole_center is None:
        return 0, 0
    
    height, width = image_shape[:2]
    mole_x, mole_y = mole_center
    
    # Parameters
    FORWARD_SPEED = 3.0
    TURN_SPEED = 2.0
    CHECK_DISTANCE = 60
    DETECTION_REGION = 40
    
    # Check for walls
    # Note: In image coordinates, y=0 is TOP, so FRONT is UPWARD (smaller y)
    
    # Check FRONT of Mole
    front_check_y = max(0, mole_y - CHECK_DISTANCE)
    front_wall = check_wall_at_position(wall_mask, mole_x, front_check_y, DETECTION_REGION)
    
    # Check RIGHT of Mole
    right_check_x = min(width-1, mole_x + CHECK_DISTANCE)
    right_wall = check_wall_at_position(wall_mask, right_check_x, mole_y, DETECTION_REGION)
    
    # RIGHT-HAND RULE DECISIONS
    if front_wall:
        # Wall in front, need to turn
        if right_wall:
            # Wall on right too, turn LEFT (sharp U-turn)
            vl = -TURN_SPEED
            vr = TURN_SPEED
        else:
            # No wall on right, turn RIGHT
            vl = TURN_SPEED
            vr = -TURN_SPEED
    else:
        # No wall in front
        if not right_wall:
            # No wall on right, curve RIGHT to find wall
            vl = FORWARD_SPEED * 1.2
            vr = FORWARD_SPEED * 0.8
        else:
            # Wall on right, follow it (go straight)
            vl = FORWARD_SPEED
            vr = FORWARD_SPEED
    
    return vl, vr, front_wall, right_wall


def display_debug_info(image, mole_bbox, mole_center, wall_mask, 
                       front_wall, right_wall, mole_vl, mole_vr, hawk_vx, hawk_vy):
    """
    Display debugging information on the image.
    """
    display_img = image.copy()
    height, width = display_img.shape[:2]
    
    # Draw Mole bounding box if detected
    if mole_bbox is not None:
        x, y, w, h = mole_bbox
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw center point
        if mole_center:
            cv2.circle(display_img, mole_center, 5, (0, 0, 255), -1)
            
            # Draw detection zones
            CHECK_DISTANCE = 60
            DETECTION_REGION = 40
            
            # Front detection zone
            front_y = max(0, mole_center[1] - CHECK_DISTANCE)
            cv2.rectangle(display_img, 
                         (mole_center[0] - DETECTION_REGION//2, front_y - DETECTION_REGION//2),
                         (mole_center[0] + DETECTION_REGION//2, front_y + DETECTION_REGION//2),
                         (0, 255, 255), 1)
            
            # Right detection zone
            right_x = min(width-1, mole_center[0] + CHECK_DISTANCE)
            cv2.rectangle(display_img,
                         (right_x - DETECTION_REGION//2, mole_center[1] - DETECTION_REGION//2),
                         (right_x + DETECTION_REGION//2, mole_center[1] + DETECTION_REGION//2),
                         (255, 255, 0), 1)
    
    # Add status text
    cv2.putText(display_img, f"Front Wall: {'YES' if front_wall else 'NO'}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Right Wall: {'YES' if right_wall else 'NO'}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Mole Vel: L={mole_vl:.1f}, R={mole_vr:.1f}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Hawk Vel: X={hawk_vx:.2f}, Y={hawk_vy:.2f}", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create composite with wall mask
    wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
    composite = np.hstack([display_img, wall_display])
    
    return composite

def control_logic(sim):
    """
    Purpose:
    ---
    Implement your solution in this function. Use the send_mole_velocity and send_hawk_velocity functions to control the bots.
    
    Input Arguments:
    ---
    `sim` : ZeroMQ RemoteAPI object
    
    Returns:
    ---
    None    
    """
    
    print("Starting Right-Wall Following Algorithm...")
    print("Algorithm: Right-hand rule for maze solving")
    
    # Initialize state
    mole_lost_count = 0
    max_lost_frames = 10
    
    # Timing control
    START_TIME = time.time()
    HAWK_HOVER_TIME = 0.0  # Hawk hovers for 1 seconds
    MOLE_STRAIGHT_TIME = 1.0  # Mole goes straight for 1. seconds
    
    # Mole straight phase flag
    mole_in_straight_phase = True
    
    while True:
        ###########################
        # WRITE YOUR CODE HERE ON #
        ###########################
        
        # Get camera image
        image = get_camera_image(sim)
        if image is None:
            time.sleep(0.05)
            continue
        
        current_time = time.time()
        elapsed_time = current_time - START_TIME
        
        # Step 1: Detect Mole
        mole_bbox, mole_center = detect_mole(image)
        
        # Step 2: Detect walls
        wall_mask = detect_walls(image)
        
        # Step 3: Control Hawk
        if elapsed_time < HAWK_HOVER_TIME:
            # Hawk hovers in place (zero velocity) for first 2 seconds
            hawk_vx = 0.0
            hawk_vy = 0.0
            hawk_mode = "HOVERING"
        else:
            # After 2 seconds, start tracking
            hawk_vx, hawk_vy = calculate_hawk_velocity(mole_center, image.shape)
            hawk_mode = "TRACKING"
        
        # Step 4: Control Mole
        if mole_center is not None:
            mole_lost_count = 0
            
            if mole_in_straight_phase and elapsed_time < MOLE_STRAIGHT_TIME:
                # Mole goes straight initially
                mole_vl = 3.0
                mole_vr = 3.0
                front_wall, right_wall = False, False
                mole_mode = "INITIAL STRAIGHT"
            else:
                # Switch to wall following
                mole_in_straight_phase = False
                mole_vl, mole_vr, front_wall, right_wall = calculate_mole_velocity(
                    mole_center, wall_mask, image.shape
                )
                mole_mode = "WALL FOLLOWING"
        else:
            mole_lost_count += 1
            if mole_lost_count > max_lost_frames:
                mole_vl, mole_vr = 0, 0
                front_wall, right_wall = False, False
                mole_mode = "STOPPED (LOST)"
            else:
                mole_vl, mole_vr = 2.0, 2.0
                front_wall, right_wall = False, False
                mole_mode = "SEARCHING"
        
        # Step 5: Send velocity commands
        send_mole_velocity(sim, mole_vl, mole_vr)
        send_hawk_velocity(sim, hawk_vx, -hawk_vy)
        
        # Step 6: Display for debugging
        debug_image = display_debug_info(
            image, mole_bbox, mole_center, wall_mask,
            front_wall, right_wall, mole_vl, mole_vr, hawk_vx, hawk_vy
        )
        
        # Add status overlay
        status_y = 30
        cv2.putText(debug_image, f"Time: {elapsed_time:.1f}s", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_image, f"Hawk: {hawk_mode}", 
                   (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if hawk_mode == "TRACKING" else (0, 255, 255), 2)
        cv2.putText(debug_image, f"Mole: {mole_mode}", 
                   (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if "FOLLOWING" in mole_mode else 
                   (0, 255, 255) if "STRAIGHT" in mole_mode else 
                   (255, 0, 0), 2)
        
        cv2.imshow('Hawk View (Camera + Wall Detection)', debug_image)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        
        time.sleep(0.05)
    
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