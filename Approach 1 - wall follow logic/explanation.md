
The file is organized into the following sections:

1. Imports and configuration  
2. Global constants and persistent variables  
3. Perception functions  
4. State and orientation handling  
5. Motion command generation  
6. Actuation helpers  
7. Main control loop  

Each section has a well-defined responsibility and does not implicitly alter the behavior of other sections.

3. Global Constants and Parameters

Global constants define fixed values used throughout execution:

- HSV thresholds for object and wall detection   such as the upper and lower bounds fot the wall detection.
- Image geometry and scaling values  
- PID gains for the drone  
- Velocity limits  
- Turn durations and stabilization times  

These parameters define system behavior but are never modified at runtime.

4. Persistent state 
The code maintains several persistent variables that represent logical state:

- Navigation phase indicators  
- Turn execution flags  
- Orientation state of the mole  
- Timing references for turns and delays  
- Orientation history and counters  

These variables represent navigation memory, not instantaneous motion.  
They change only at explicit transition points in the code.

5. Wall Detection

The wall detection function:

- Uses dual HSV masks to handle hue wrap-around
- Combines the masks into a single binary image
- Applies morphological cleanup

The resulting mask represents wall locations for the current frame only.

6. Local Wall Queries

Wall presence is checked in localized regions relative to the detected mole position:

- Front
- Left
- Right

Each query evaluates pixel density within a defined window and returns a boolean value.  
These checks do not modify any state.

7. Orientation Tracking and Reference Frames (ACTUAL MECHANISM ON HOW WE KEEP TRACK OF THE TURNS AND CHANEG THE VELOCITY INOUTS TO THE MOTOR)

### 7.1 Orientation Representation

The mole’s orientation is stored as a discrete state representing completed turns:

- `0` : initial orientation  
- `1` : right turn completed  
- `-1` : left turn completed  
- `2` : U-turn completed  

This value does **not** represent current velocity or heading at a given instant.  
It represents how the mole’s local frame has rotated relative to its initial frame.


### 7.2 When Orientation Changes

Orientation is updated only when a turn finishes.  
It is not updated during:

- Straight motion  
- Curved motion  
- Turn execution  

A time-based debounce ensures that each turn updates orientation exactly once.

---

### 7.3 Frame-of-Reference Difference Between Hawk and Mole

The mole and the hawk operate in **different reference frames**:

- The **mole** moves in its own local frame, which rotates when it turns.
- The **hawk** can only move in the simulator’s global **X–Y plane**.

Because of this difference, the same logical motion relative to the mole may require **different velocity components** in the hawk’s frame.

---

### 7.4 Velocity Transformation Based on Orientation

When the mole changes orientation, the hawk’s velocity commands are adjusted to compensate for the rotated frame.

For example:

- A forward-following motion in the mole’s frame may require  
  `(-vx, vy)` instead of `(vx, vy)` in the hawk’s frame
- This sign change is not arbitrary; it reflects the rotated relationship
  between the mole’s heading and the global X–Y axes

The **state does not reset** during this process.  
Only the **velocity components change** to match the new frame alignment.

8. Velocity Generation

Velocity profiles are defined for:

- Straight motion  
- Right turn  
- Left turn  
- U-turn  
- Gentle curvature  

These functions return velocity values only.  
They do not update orientation, state flags, or timing variable


9. Mole Control Logic

The mole control function computes wheel velocities by progressing through distinct phases:

1. Initial straight motion  
2. Active turn execution  
3. Post-turn straight stabilization  
4. Normal wall-based decision making  

Only one phase is active at any time.

---

### 9.1 Initial Motion Phase

During startup:

- The mole moves straight
- Orientation is initialized once
- No turn decisions are evaluated

---

### 9.2 Turn Execution Phase

Once a turn begins:

- Velocity is locked to the selected turn profile
- The turn runs for a fixed duration
- No new decisions are allowed

Orientation remains unchanged during this phase.

---

### 9.3 Turn Completion

At the end of the turn:

- Orientation is updated
- Turn flags are cleared
- A short straight-motion stabilization phase begins

---

### 9.4 Post-Turn Straight Phase

During stabilization:

- The mole moves straight
- Wall checks are temporarily ignored
- Orientation remains fixed

This prevents immediate re-triggering of turns.

---

### 9.5 Normal Decision Phase

When no special phase is active:

- Wall presence is evaluated
- Decisions follow a right-preference rule
- New turns may be initiated

State changes occur only when a turn begins or completes.

---

## 10. Drone Control Logic

The hawk’s velocity is computed independently:

- A startup delay prevents early motion
- PID control centers the detected mole
- Velocity is clamped to safe limits
- Velocity components are transformed based on current orientation state

The hawk does not modify mole state or navigation memory.

---

## 11. Visualization

The visualization overlays:

- Detected positions
- Wall-check regions
- Current state and timers
- Orientation and turn history

Visualization reads state but does not modify it.

---

## 12. Actuation Helpers

Helper functions send velocity values to the simulator for:

- Mole wheel velocities
- Hawk planar velocities

These functions perform no computation or validation.

---

## 13. Main Control Loop

The main loop repeatedly:

1. Captures camera input  
2. Runs perception  
3. Computes velocities  
4. Sends commands  
5. Updates visualization  
