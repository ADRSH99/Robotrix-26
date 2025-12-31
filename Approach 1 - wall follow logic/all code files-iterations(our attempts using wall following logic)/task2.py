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
    Vision-based navigation logic.
    """
    # Mole (Rover) parameters
    BASE_SPEED = 2.0
    TURN_SPEED = 2.0
    STOP_DISTANCE_THRESHOLD = 35
    
    # Hawk (Drone) parameters
    TARGET_AREA = 2500
    AREA_GAIN = 0.001
    CENTER_GAIN = 0.0025
    MIN_AREA = 500
    
    # PID for smoother control
    prev_err_x = 0
    integral_x = 0
    
    while True:
        frame = get_camera_image(sim)
        if frame is None:
            continue

        h, w, _ = frame.shape
        cx_img = w // 2
        cy_img = h // 2
        
        # Draw center lines for reference
        cv2.line(frame, (cx_img, 0), (cx_img, h), (100, 100, 100), 1)
        cv2.line(frame, (0, cy_img), (w, cy_img), (100, 100, 100), 1)

        # -------------------- ENHANCED MOLE DETECTION --------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjusted yellow color range based on image
        # Yellow appears bright in the image
        lower_yellow = np.array([15, 80, 80])   # Lower saturation/value
        upper_yellow = np.array([40, 255, 255])  # Broader hue range
        
        # Create mask for yellow
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Also look for bright areas (the mole is very bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_yellow, mask_bright)
        
        # Morphological operations to clean up
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area to remove noise
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_AREA:  # Filter small noise
                    valid_contours.append(contour)
            
            if valid_contours:
                # Find the largest valid contour
                c = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                
                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(c)
                cx = x + w_box // 2
                cy = y + h_box // 2
                bbox_area = w_box * h_box
                
                # Draw detection results
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw center point
                cv2.circle(frame, (cx_img, cy_img), 5, (255, 0, 0), -1)
                
                # Draw line between centers
                cv2.line(frame, (cx_img, cy_img), (cx, cy), (255, 255, 0), 2)
                
                # -------------------- WALL DETECTION --------------------
                # Use color-based wall detection (walls appear gray/dark)
                # Convert to HSV for better color separation
                lower_wall = np.array([0, 0, 0])
                upper_wall = np.array([180, 50, 150])  # Dark, low saturation
                mask_wall = cv2.inRange(hsv, lower_wall, upper_wall)
                
                # Define front region (where rover would hit)
                front_height = int(0.4 * h)  # Lower 40%
                front_width_start = int(0.3 * w)
                front_width_end = int(0.7 * w)
                
                front_region = mask_wall[h - front_height:h, front_width_start:front_width_end]
                front_density = np.mean(front_region) / 2.55  # Scale 0-255 to 0-100
                
                # Draw front region
                cv2.rectangle(frame, (front_width_start, h - front_height), 
                            (front_width_end, h), (255, 0, 255), 2)
                
                # -------------------- MOLE CONTROL --------------------
                err_x = cx - cx_img
                
                # Simple PID for turning
                integral_x = integral_x * 0.9 + err_x  # Leaky integrator
                derivative = err_x - prev_err_x
                prev_err_x = err_x
                
                # Wall avoidance logic
                if front_density > STOP_DISTANCE_THRESHOLD:
                    # Wall detected - stop and turn
                    if front_density > 50:  # Very close wall
                        vl, vr = 0.0, 0.0  # Complete stop
                        time.sleep(0.5)    # Brief pause
                        # Turn away from wall
                        vl, vr = -TURN_SPEED, TURN_SPEED  # Default left turn
                    else:
                        # Slow down and prepare to turn
                        vl = BASE_SPEED * 0.3
                        vr = BASE_SPEED * 0.3
                        
                else:
                    # No wall threat, follow mole
                    if abs(err_x) > 30:
                        # Calculate turn based on error
                        turn = CENTER_GAIN * (err_x + 0.1 * integral_x + 0.05 * derivative)
                        turn = max(min(turn, 1.5), -1.5)
                        
                        vl = BASE_SPEED - turn
                        vr = BASE_SPEED + turn
                    else:
                        # Centered, move forward
                        vl, vr = BASE_SPEED, BASE_SPEED
                
                # Ensure reasonable speeds
                vl = max(min(vl, 3.0), -3.0)
                vr = max(min(vr, 3.0), -3.0)
                
                send_mole_velocity(sim, vl, vr)
                
                # -------------------- HAWK CONTROL --------------------
                # Error calculations
                area_error = TARGET_AREA - bbox_area
                x_error = cx - cx_img
                y_error = cy - cy_img
                
                # Control logic for drone:
                # 1. Maintain mole in center (x direction)
                # 2. Maintain distance (z/forward direction based on area)
                # 3. Keep some altitude (y position)
                
                # Forward/backward (z) - based on area
                vx = AREA_GAIN * area_error
                
                # Left/right (x) - center mole horizontally
                vy = -CENTER_GAIN * x_error
                
                # Add some upward bias to maintain altitude
                vz = 0.01  # Small upward bias
                
                # Limit velocities
                vx = max(min(vx, 0.15), -0.15)  # Slow forward/backward
                vy = max(min(vy, 0.2), -0.2)    # Moderate sideways
                
                # If mole is getting too small (far away), move forward gently
                if bbox_area < 800:
                    vx = 0.1  # Gentle forward
                elif bbox_area > 4000:
                    vx = -0.1  # Gentle backward
                
                send_hawk_velocity(sim, vx, vy)
                
                # Display information
                cv2.putText(frame, f"Mole Area: {bbox_area}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Front Wall: {front_density:.1f}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Rover: {vl:.1f}, {vr:.1f}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Hawk: {vx:.2f}, {vy:.2f}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Error X: {err_x}", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            else:
                # No valid contours found
                cv2.putText(frame, "No valid mole detected", (w//2 - 100, h//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                send_mole_velocity(sim, -1.0, 1.0)  # Rotate to search
                send_hawk_velocity(sim, 0.0, 0.0)   # Hover
        else:
            # No contours at all
            cv2.putText(frame, "SEARCHING...", (w//2 - 80, h//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            send_mole_velocity(sim, -1.5, 1.5)  # Rotate faster
            send_hawk_velocity(sim, 0.0, 0.0)   # Hover
        
        # Display mask for debugging
        mask_resized = cv2.resize(mask, (w//4, h//4))
        mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        frame[10:10+h//4, 10:10+w//4] = mask_bgr
        
        cv2.imshow("Hawk Camera - Mole Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Pause on spacebar
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == ord(' '):
                    break
        
        time.sleep(0.05)
    
    cv2.destroyAllWindows()

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