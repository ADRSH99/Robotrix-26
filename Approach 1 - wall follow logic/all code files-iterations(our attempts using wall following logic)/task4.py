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
def control_logic(sim):
    """
    Simple control logic: Move forward, turn when wall detected.
    Drone stays above rover by matching movements.
    """
    
    # Constants
    FORWARD_SPEED = 1.5
    TURN_SPEED = 1.2
    WALL_THRESHOLD = 30  # Lower threshold to prevent false positives
    
    # State
    is_turning = False
    turn_direction = 1  # 1 for right, -1 for left
    turn_start_time = 0
    turn_duration = 2.0  # seconds to turn
    
    while True:
        frame = get_camera_image(sim)
        if frame is None:
            time.sleep(0.05)
            continue
        
        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2
        
        # -------------------- SIMPLE WALL DETECTION --------------------
        # Convert to grayscale for simple wall detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding - walls are darker
        _, wall_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Look only at the very bottom center for immediate walls
        bottom_height = int(0.15 * h)  # Only bottom 15%
        bottom_width = int(0.3 * w)    # Center 30%
        bottom_start = int(0.35 * w)
        
        front_region = wall_mask[h - bottom_height:h, bottom_start:bottom_start + bottom_width]
        
        # Calculate wall density
        wall_density = np.mean(front_region) if front_region.size > 0 else 0
        wall_detected = wall_density > WALL_THRESHOLD
        
        # Draw detection zone
        cv2.rectangle(frame, (bottom_start, h - bottom_height), 
                     (bottom_start + bottom_width, h), (0, 255, 255), 2)
        
        # -------------------- ROVER DETECTION --------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        rover_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(rover_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rover_detected = False
        rover_x = center_x
        rover_y = center_y
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 100:
                rover_detected = True
                x, y, w_box, h_box = cv2.boundingRect(largest)
                rover_x = x + w_box // 2
                rover_y = y + h_box // 2
                
                # REQUIRED: Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.circle(frame, (rover_x, rover_y), 5, (0, 0, 255), -1)
        
        # -------------------- SIMPLE CONTROL LOGIC --------------------
        current_time = time.time()
        
        if is_turning:
            # Currently turning
            if current_time - turn_start_time < turn_duration:
                # Continue turning
                vl = -turn_direction * TURN_SPEED
                vr = turn_direction * TURN_SPEED
            else:
                # Finished turning
                is_turning = False
                vl = FORWARD_SPEED
                vr = FORWARD_SPEED
        else:
            # Moving forward
            if wall_detected:
                # Wall detected - start turning
                is_turning = True
                turn_start_time = current_time
                # Default turn direction: right
                turn_direction = 1
                vl = -turn_direction * TURN_SPEED  # Start turning immediately
                vr = turn_direction * TURN_SPEED
            else:
                # No wall - keep moving forward
                vl = FORWARD_SPEED
                vr = FORWARD_SPEED
        
        # Send rover velocity
        send_mole_velocity(sim, vl, vr)
        
        # -------------------- SIMPLE DRONE CONTROL --------------------
        # Drone should stay above rover
        # Simple approach: If rover is detected, adjust to center it
        
        if rover_detected:
            # Calculate how far rover is from center
            error_x = rover_x - center_x
            error_y = rover_y - center_y
            
            # Adjust drone position to center rover
            # Small gains for gentle movement
            vy = -0.001 * error_x  # Horizontal adjustment
            vx = 0.001 * error_y   # Forward/backward adjustment
            
            # If rover is moving forward, drone should also move slightly forward
            if not is_turning and vl > 0 and vr > 0:
                vx += 0.05  # Slight forward movement to keep up
        else:
            # Rover lost - hover in place
            vx = 0.0
            vy = 0.0
        
        # If turning, add sideways movement for drone too
        if is_turning:
            vy += turn_direction * 0.08
        
        # Limit drone speeds
        vx = max(min(vx, 0.1), -0.1)
        vy = max(min(vy, 0.1), -0.1)
        
        # Send drone velocity
        send_hawk_velocity(sim, vx, vy)
        
        # -------------------- DISPLAY --------------------
        # Status text
        cv2.putText(frame, f"State: {'TURNING' if is_turning else 'FORWARD'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Wall: {wall_detected} ({wall_density:.0f})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Rover: {rover_detected}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Rover Vel: {vl:.1f}, {vr:.1f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Drone Vel: {vx:.2f}, {vy:.2f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Center marker
        cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)
        
        # Show wall mask in corner
        wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        wall_display = cv2.resize(wall_display, (w//4, h//4))
        frame[10:10+h//4, w-w//4-10:w-10] = wall_display
        
        cv2.imshow("Hawk Camera - Simple Navigation", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to force forward
            is_turning = False
            vl = FORWARD_SPEED
            vr = FORWARD_SPEED
            send_mole_velocity(sim, vl, vr)
        
        time.sleep(0.05)
    
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