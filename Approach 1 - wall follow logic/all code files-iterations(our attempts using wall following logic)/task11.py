'''
========================================================================================
IEEE Robotrix 2025-26 – Final Hackathon
Team: ROBODIH
========================================================================================
'''

####################### IMPORT MODULES #######################
import sys, traceback, time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
##############################################################

######################## CONSTANTS ###########################

YELLOW_LOWER = np.array([20,100,100])
YELLOW_UPPER = np.array([35,255,255])

WALL_LOWER_1 = np.array([0,150,150])
WALL_UPPER_1 = np.array([10,255,255])
WALL_LOWER_2 = np.array([170,150,150])
WALL_UPPER_2 = np.array([180,255,255])

IMAGE_CENTER = (256,256)

# ---- TURN & MOTION (TUNED) ----
MOLE_BASE_SPEED = 2.0
MOLE_TURN_SPEED = 1.4        # ↓ reduced
TURN_DURATION   = 0.65       # ↓ reduced
STRAIGHT_AFTER_TURN = 2.5

INITIAL_STRAIGHT_TIME = 2.0
HAWK_DELAY_TIME = 1.0

DETECTION_DISTANCE = 60
CHECK_REGION_SIZE = 30

# ---- DRONE PID ----
DRONE_KP = 0.007
DRONE_KD = 0.023

##############################################################

######################## STATE ################################

HAWK_ORIENTATION = 0  # 0↑ 1→ 2↓ 3←
MOLE_TURN_COUNT = 0

pid = dict(ix=0, iy=0, px=0, py=0)

start_time = 0
in_turn = False
turn_type = None
turn_start = 0

in_post_turn = False
post_turn_start = 0

##############################################################

def detect_mole(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,YELLOW_LOWER,YELLOW_UPPER)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    (x,y),r = cv2.minEnclosingCircle(max(cnts,key=cv2.contourArea))
    return (int(x),int(y)) if r > 5 else None

def detect_walls(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv,WALL_LOWER_1,WALL_UPPER_1)
    m2 = cv2.inRange(hsv,WALL_LOWER_2,WALL_UPPER_2)
    return cv2.bitwise_or(m1,m2)

def check_wall(mask,x,y,dir):
    h,w = mask.shape
    offsets = {
        "front":[(0,-DETECTION_DISTANCE),(DETECTION_DISTANCE,0),(0,DETECTION_DISTANCE),(-DETECTION_DISTANCE,0)],
        "right":[(DETECTION_DISTANCE,0),(0,DETECTION_DISTANCE),(-DETECTION_DISTANCE,0),(0,-DETECTION_DISTANCE)],
        "left" :[(-DETECTION_DISTANCE,0),(0,-DETECTION_DISTANCE),(DETECTION_DISTANCE,0),(0,DETECTION_DISTANCE)]
    }
    dx,dy = offsets[dir][HAWK_ORIENTATION]
    cx,cy = np.clip(x+dx,0,w-1), np.clip(y+dy,0,h-1)
    sx,ex = max(0,cx-15),min(w,cx+15)
    sy,ey = max(0,cy-15),min(h,cy+15)
    return np.count_nonzero(mask[sy:ey,sx:ex]) > 0.25*(30*30)

def update_orientation(turn):
    global HAWK_ORIENTATION, MOLE_TURN_COUNT
    if turn=="RIGHT": HAWK_ORIENTATION=(HAWK_ORIENTATION+1)%4
    elif turn=="LEFT": HAWK_ORIENTATION=(HAWK_ORIENTATION-1)%4
    elif turn=="U_TURN": HAWK_ORIENTATION=(HAWK_ORIENTATION+2)%4
    MOLE_TURN_COUNT+=1

def compute_mole_velocity(pos, mask, t):
    global in_turn, turn_type, turn_start
    global in_post_turn, post_turn_start

    if t-start_time < INITIAL_STRAIGHT_TIME:
        return MOLE_BASE_SPEED,MOLE_BASE_SPEED

    if pos is None:
        return 0,0

    if in_turn:
        if t-turn_start < TURN_DURATION:
            if turn_type=="RIGHT": return MOLE_TURN_SPEED,-MOLE_TURN_SPEED
            if turn_type=="LEFT": return -MOLE_TURN_SPEED,MOLE_TURN_SPEED
            return -MOLE_TURN_SPEED,MOLE_TURN_SPEED
        in_turn=False
        in_post_turn=True
        post_turn_start=t

    if in_post_turn:
        if t-post_turn_start < STRAIGHT_AFTER_TURN:
            return MOLE_BASE_SPEED,MOLE_BASE_SPEED
        in_post_turn=False
        update_orientation(turn_type)

    fw = check_wall(mask,pos[0],pos[1],"front")
    rw = check_wall(mask,pos[0],pos[1],"right")
    lw = check_wall(mask,pos[0],pos[1],"left")

    if fw:
        in_turn=True
        turn_start=t
        turn_type = "RIGHT" if not rw else "LEFT" if not lw else "U_TURN"
        return 0,0

    if not rw:
        return MOLE_BASE_SPEED,MOLE_BASE_SPEED*0.85
    return MOLE_BASE_SPEED,MOLE_BASE_SPEED

def compute_hawk_velocity(pos):
    if pos is None:
        pid.update(ix=0,iy=0,px=0,py=0)
        return 0,0

    ex,ey = pos[0]-256,pos[1]-256
    if HAWK_ORIENTATION==1: ex,ey=ey,-ex
    if HAWK_ORIENTATION==2: ex,ey=-ex,-ey
    if HAWK_ORIENTATION==3: ex,ey=-ey,ex

    vx = DRONE_KP*ex + DRONE_KD*(ex-pid["px"])
    vy = DRONE_KP*ey + DRONE_KD*(ey-pid["py"])
    pid["px"],pid["py"]=ex,ey

    # ---- SAFETY CLAMP DURING MOLE TURN ----
    if in_turn:
        vx*=0.3
        vy*=0.3

    return np.clip(vx,-4,4),np.clip(-vy,-4,4)

def get_camera_image(sim):
    p=sim.getBufferSignal('hawk_image')
    if p is None: return None
    x,y=sim.getInt32Signal('hawk_res_x'),sim.getInt32Signal('hawk_res_y')
    img=np.frombuffer(p,np.uint8).reshape((y,x,3))
    return cv2.flip(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),0)

def send_mole_velocity(sim,vl,vr):
    sim.setFloatSignal('mole_left_vel',float(vl))
    sim.setFloatSignal('mole_right_vel',float(vr))

def send_hawk_velocity(sim,vx,vy):
    sim.setFloatSignal('hawk_vx',float(vx))
    sim.setFloatSignal('hawk_vy',float(vy))

def control_logic(sim):
    global start_time
    start_time=time.time()
    while True:
        img=get_camera_image(sim)
        if img is None: continue
        pos=detect_mole(img)
        mask=detect_walls(img)
        vl,vr=compute_mole_velocity(pos,mask,time.time())
        vx,vy=compute_hawk_velocity(pos)
        send_mole_velocity(sim,vl,vr)
        send_hawk_velocity(sim,vx,vy)
        if cv2.waitKey(1)==27: break
        time.sleep(0.03)

##############################################################

if __name__=="__main__":
    client=RemoteAPIClient()
    sim=client.getObject('sim')
    sim.startSimulation()
    control_logic(sim)
    sim.stopSimulation()
