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

class MazeSolver:
    """Main controller class for maze navigation"""
    
    def __init__(self):
        # System states
        self.state = "INITIAL"
        self.start_time = time.time()
        
        # Velocity parameters
        self.mole_slow_speed = 1.0
        self.mole_normal_speed = 2.0
        self.hawk_slow_speed = 0.1
        self.hawk_normal_speed = 0.2
        
        # Timing parameters
        self.initial_mole_time = 2.0  # Mole goes straight for 2 seconds
        self.initial_hawk_time = 5.0  # Hawk waits 3 seconds before tracking
        
        # Tracking history
        self.mole_positions = []
        self.mole_lost_counter = 0
        self.max_lost_frames = 15
        
        # Wall following parameters
        self.wall_follow_started = False
        
    def detect_mole_simple(self, image):
        """Simple yellow detection for Mole"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow color range for Mole
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:  # Minimum size
                x, y, w, h = cv2.boundingRect(largest)
                center_x = x + w // 2
                center_y = y + h // 2
                return (center_x, center_y), (x, y, w, h), mask
        
        return None, None, mask
    
    def detect_walls_simple(self, image):
        """Simple red wall detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color ranges (walls are red)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        wall_mask = cv2.bitwise_or(mask1, mask2)
        
        # Dilate to make walls more visible
        kernel = np.ones((7,7), np.uint8)
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
        
        return wall_mask
    
    def check_wall_proximity(self, wall_mask, mole_center, direction="front", distance=40):
        """Check for walls in specific directions from Mole"""
        if mole_center is None:
            return False
        
        height, width = wall_mask.shape
        x, y = mole_center
        
        # Define check regions based on direction
        if direction == "front":
            # Check in front of Mole (upwards in image)
            check_y = max(0, y - distance)
            region_y1 = max(0, check_y - 20)
            region_y2 = max(0, check_y + 20)
            region_x1 = max(0, x - 30)
            region_x2 = min(width, x + 30)
            
        elif direction == "right":
            # Check to the right of Mole
            check_x = min(width-1, x + distance)
            region_x1 = max(0, check_x - 20)
            region_x2 = min(width, check_x + 20)
            region_y1 = max(0, y - 30)
            region_y2 = min(height, y + 30)
            
        elif direction == "left":
            # Check to the left of Mole
            check_x = max(0, x - distance)
            region_x1 = max(0, check_x - 20)
            region_x2 = min(width, check_x + 20)
            region_y1 = max(0, y - 30)
            region_y2 = min(height, y + 30)
        else:
            return False
        
        # Extract and check region
        if region_x1 < region_x2 and region_y1 < region_y2:
            region = wall_mask[region_y1:region_y2, region_x1:region_x2]
            wall_pixels = np.sum(region) / 255
            
            # If more than 15% of region has walls, return True
            region_area = (region_x2 - region_x1) * (region_y2 - region_y1)
            return wall_pixels > (0.15 * region_area)
        
        return False
    
    def calculate_hawk_tracking(self, mole_center, image_shape, current_time):
        """Calculate Hawk velocity for tracking Mole"""
        if mole_center is None:
            return 0.0, 0.0, "LOST"
        
        height, width = image_shape
        center_x, center_y = width // 2, height // 2
        mole_x, mole_y = mole_center
        
        # Calculate error from center
        error_x = mole_x - center_x
        error_y = mole_y - center_y
        
        # Check if initial waiting period is over
        elapsed = current_time - self.start_time
        if elapsed < self.initial_hawk_time:
            return 0.0, 0.0, "WAITING"
        
        # Gentle proportional control with deadzone
        deadzone = 80  # Pixels
        
        vx, vy = 0.0, 0.0
        
        if abs(error_x) > deadzone:
            vx = -0.001 * error_x  # Very slow movement
        
        if abs(error_y) > deadzone:
            vy = -0.001 * error_y
        
        # Limit maximum speed
        max_speed = self.hawk_normal_speed
        vx = max(-max_speed, min(max_speed, vx))
        vy = max(-max_speed, min(max_speed, vy))
        
        # Very small minimum threshold to avoid drift
        min_threshold = 0.01
        if abs(vx) < min_threshold:
            vx = 0.0
        if abs(vy) < min_threshold:
            vy = 0.0
        
        return vx, vy, "TRACKING"
    
    def calculate_mole_navigation(self, mole_center, wall_mask, image_shape, current_time):
        """Calculate Mole velocity for maze navigation"""
        if mole_center is None:
            return 0.0, 0.0, "LOST", False, False
        
        elapsed = current_time - self.start_time
        
        # Initial straight movement
        if elapsed < self.initial_mole_time:
            return self.mole_slow_speed, self.mole_slow_speed, "INITIAL_STRAIGHT", False, False
        
        # After initial phase, start wall following
        if not self.wall_follow_started:
            self.wall_follow_started = True
            print("Starting wall-following navigation")
        
        # Check walls in different directions
        front_wall = self.check_wall_proximity(wall_mask, mole_center, "front", 50)
        right_wall = self.check_wall_proximity(wall_mask, mole_center, "right", 40)
        left_wall = self.check_wall_proximity(wall_mask, mole_center, "left", 40)
        
        # Right-hand rule navigation
        if front_wall:
            # Wall in front - need to turn
            if not right_wall:
                # No wall on right, turn right
                vl = self.mole_slow_speed * 0.5
                vr = -self.mole_slow_speed * 0.5
                return vl, vr, "TURNING_RIGHT", front_wall, right_wall
            elif not left_wall:
                # Wall on right but not left, turn left
                vl = -self.mole_slow_speed * 0.5
                vr = self.mole_slow_speed * 0.5
                return vl, vr, "TURNING_LEFT", front_wall, right_wall
            else:
                # Dead end, turn around
                vl = -self.mole_slow_speed * 0.7
                vr = self.mole_slow_speed * 0.7
                return vl, vr, "TURNING_AROUND", front_wall, right_wall
        else:
            # No wall in front
            if not right_wall:
                # No wall on right, curve slightly right
                vl = self.mole_normal_speed * 1.1
                vr = self.mole_normal_speed * 0.9
                return vl, vr, "CURVING_RIGHT", front_wall, right_wall
            else:
                # Wall on right, follow it straight
                vl = self.mole_normal_speed
                vr = self.mole_normal_speed
                return vl, vr, "FOLLOWING_WALL", front_wall, right_wall
    
    def update_state_machine(self, mole_center, current_time):
        """Update the system state machine"""
        elapsed = current_time - self.start_time
        
        if self.state == "INITIAL":
            if elapsed > self.initial_mole_time:
                self.state = "TRACKING"
                print(f"State changed to TRACKING at {elapsed:.1f}s")
        
        elif self.state == "TRACKING":
            if mole_center is None:
                self.mole_lost_counter += 1
                if self.mole_lost_counter > self.max_lost_frames:
                    self.state = "RECOVERY"
                    print("State changed to RECOVERY (Mole lost)")
            else:
                self.mole_lost_counter = 0
                if elapsed > self.initial_hawk_time + 5.0:  # After tracking for 5 seconds
                    self.state = "NAVIGATING"
                    print("State changed to NAVIGATING")
        
        elif self.state == "NAVIGATING":
            # Main navigation state
            pass
        
        elif self.state == "RECOVERY":
            # Try to recover lost Mole
            if mole_center is not None:
                self.state = "TRACKING"
                print("State changed back to TRACKING (Mole found)")
    
    def draw_debug_info(self, image, mole_center, mole_bbox, wall_mask, 
                       mole_vl, mole_vr, hawk_vx, hawk_vy, mole_status, hawk_status):
        """Draw debugging information on image"""
        display = image.copy()
        height, width = display.shape[:2]
        
        # Draw Mole if detected
        if mole_bbox is not None:
            x, y, w, h = mole_bbox
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if mole_center:
                cv2.circle(display, mole_center, 5, (0, 0, 255), -1)
                
                # Draw detection zones
                cv2.circle(display, (mole_center[0], max(0, mole_center[1] - 50)), 
                          8, (255, 255, 0), 1)  # Front
                cv2.circle(display, (min(width-1, mole_center[0] + 40), mole_center[1]), 
                          8, (0, 255, 255), 1)  # Right
                cv2.circle(display, (max(0, mole_center[0] - 40), mole_center[1]), 
                          8, (255, 0, 255), 1)  # Left
        
        # Add status information
        elapsed = time.time() - self.start_time
        
        cv2.putText(display, f"State: {self.state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Time: {elapsed:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Mole: {mole_status}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Hawk: {hawk_status}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, f"Vel - Mole: ({mole_vl:.1f}, {mole_vr:.1f})", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Vel - Hawk: ({hawk_vx:.3f}, {hawk_vy:.3f})", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Create composite with wall mask
        wall_display = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        composite = np.hstack([display, wall_display])
        
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
    
    print("=" * 60)
    print("MAZE NAVIGATION SYSTEM INITIALIZING...")
    print("Strategy: Right-hand wall following")
    print("Phases: 1) Initial straight 2) Hawk tracking 3) Wall following")
    print("=" * 60)
    
    # Initialize maze solver
    solver = MazeSolver()
    
    try:
        while True:
            # Get camera image
            image = get_camera_image(sim)
            if image is None:
                time.sleep(0.02)
                continue
            
            current_time = time.time()
            
            # Step 1: Detect Mole and walls
            mole_center, mole_bbox, mole_mask = solver.detect_mole_simple(image)
            wall_mask = solver.detect_walls_simple(image)
            
            # Step 2: Update state machine
            solver.update_state_machine(mole_center, current_time)
            
            # Step 3: Calculate Hawk velocity
            hawk_vx, hawk_vy, hawk_status = solver.calculate_hawk_tracking(
                mole_center, image.shape[:2], current_time
            )
            
            # Step 4: Calculate Mole velocity
            mole_vl, mole_vr, mole_status, front_wall, right_wall = solver.calculate_mole_navigation(
                mole_center, wall_mask, image.shape[:2], current_time
            )
            
            # Step 5: Send velocity commands (VERY IMPORTANT: Hawk at low speeds)
            send_mole_velocity(sim, mole_vl, mole_vr)
            
            # Apply additional slowing for Hawk if needed
            if abs(hawk_vx) > 0.3 or abs(hawk_vy) > 0.3:
                hawk_vx *= 0.5
                hawk_vy *= 0.5
            
            send_hawk_velocity(sim, hawk_vx, -hawk_vy)
            
            # Step 6: Display debugging information
            debug_image = solver.draw_debug_info(
                image, mole_center, mole_bbox, wall_mask,
                mole_vl, mole_vr, hawk_vx, hawk_vy, mole_status, hawk_status
            )
            
            cv2.imshow('Maze Navigation System', debug_image)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("\nSimulation stopped by user")
                break
            
            # Small delay to prevent overwhelming the simulation
            time.sleep(0.03)
            
    except Exception as e:
        print(f"\n[ERROR] Exception in control logic: {e}")
        traceback.print_exc()
    
    finally:
        # Ensure we stop both bots when exiting
        send_mole_velocity(sim, 0, 0)
        send_hawk_velocity(sim, 0, 0)
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