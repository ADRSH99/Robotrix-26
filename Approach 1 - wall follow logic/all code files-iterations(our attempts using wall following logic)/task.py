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

    BASE_SPEED = 3.0
    TURN_SPEED = 2.5

    TURN_GAIN = 0.004
    YAW_GAIN = 0.002
    TARGET_AREA = 2500

    while True:

        frame = get_camera_image(sim)
        if frame is None:
            continue

        h, w, _ = frame.shape
        cx_img = w // 2

        # -------------------- MOLE DETECTION --------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 120, 120])
        upper_yellow = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)
            cx = x + w_box // 2
            bbox_area = w_box * h_box

            # Draw bounding box (MANDATORY FOR VIDEO)
            cv2.rectangle(
                frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2
            )

            # -------------------- WALL DETECTION --------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            left_region = edges[:, :int(0.30 * w)]
            right_region = edges[:, int(0.70 * w):w]
            front_region = edges[:, int(0.45 * w):int(0.55 * w)]

            left_density = np.mean(left_region)
            right_density = np.mean(right_region)
            front_density = np.mean(front_region)

            # -------------------- MOLE CONTROL --------------------
            # -------------------- MOLE CONTROL --------------------
            err_x = cx - cx_img

            if front_density > 30:
            # FORCE TURN
                vl, vr = -TURN_SPEED, TURN_SPEED
            elif right_density < 25:
                vl, vr = TURN_SPEED, -TURN_SPEED
            else:
                vl, vr = BASE_SPEED, BASE_SPEED

            send_mole_velocity(sim, vl, vr)


            # -------------------- HAWK TRACKING and FOLLOWING --------------------
            area_error = TARGET_AREA - bbox_area
            #vx = 0.001 * area_error
            vy = -YAW_GAIN * err_x
            if abs(err_x) < 20:
                vx = 0.001 * area_error
            else:
                vx = 0.0
            vx = max(min(vx, 0.3), -0.3)
            vy = max(min(vy, 0.3), -0.3)
            send_hawk_velocity(sim, vx, vy)

        else:
            # Lost Mole → slow rotate
            send_mole_velocity(sim, 0.0, 0.0)
            send_hawk_velocity(sim, 0.0, 0.2)

        cv2.imshow("Hawk Camera", frame)
        cv2.waitKey(1)
        time.sleep(0.05)

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