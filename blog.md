# How We Won IEEE Robotrix 2025-26

> A blind robot, a drone with a camera, and the constraint that broke our first approach.

---

## The Problem

IEEE Robotrix gave us a challenge that sounds simple until you read the constraints:

**Guide a ground rover through a maze using only a drone's camera feed.**

The rover—called "The Mole"—has no sensors. No lidar. No IMU. Nothing. It's completely blind.

The drone—"The Hawk"—hovers above with a downward-facing camera. 512×512 pixels. That's your only window into the world.

And here's the kicker: **you can't ask the simulator for any position data.** The function `sim.getObjectPosition()` is banned. No global coordinates. No cheating.

You have to figure out where the rover is, which way it's facing, and how to navigate—all from raw pixels.

---

## Why Wall Following Doesn't Just Work

The obvious approach is the right-hand rule: keep your right hand on the wall, and you'll eventually reach the exit.

We tried it. Here's what goes wrong.

When you detect walls from a top-down camera, you're looking at fixed regions of the image. "Front" means pixels above the rover. "Right" means pixels to the right.

But when the rover turns 90°, its "front" is now what used to be "right" in the image. Your wall detection is suddenly checking the wrong direction.

"No problem," you think. "Just track which way the rover has turned and rotate your detection accordingly."

Now you need to maintain orientation state. And coordinate transforms between the rover's local frame and the camera's global frame. And handle accumulated errors when your timed turns aren't exactly 90°.

This is where it gets hairy. 18 iterations and 800+ lines of code later, it still failed at corners. The coordinate transforms had sign errors we couldn't track down. The rover would work for two turns, then drive straight into a wall.

---

## The Insight That Changed Everything

What if we didn't need to know which way the rover is facing?

The problem with orientation tracking is that it requires maintaining state across frames. Every frame depends on what happened before. Errors accumulate.

What if every frame was independent?

Here's the idea: instead of discrete wall detection + turn decisions, generate a **continuous path to follow**. Like a line painted on the floor. The rover doesn't need to know north from south—it just follows the line.

---

## Virtual Guide Rails

Every frame, we:

1. **Detect the walls** using HSV color segmentation (they're red)
2. **Find the wall to our right** by sampling a point 100 pixels to the rover's right side and finding the nearest wall contour
3. **Generate a guide rail** by dilating the wall, then eroding it (this creates smooth curves at corners), and extracting the edge

The result: a yellow line that runs parallel to the right wall, curving naturally around corners. No sharp 90° angles. No jumps when the wall changes.

```python
# The morphological magic
dilated = cv2.dilate(wall_mask, kernel)
filleted = cv2.erode(dilated, kernel) 
# Sharp corners become smooth curves
```

---

## Following the Line

Now the rover is just a line-follower. Classic robotics.

We place 5 virtual "IR sensors" ahead of the rover in a reverse-U pattern:

```
          [C]      
       [L]  [R]
      /        \
   [LL]        [RR]
           |
         ROVER
```

Each sensor samples the guide rail image. We weight them: far-left is +2.3, far-right is -2.3, center is 0. The weighted sum becomes the error signal for a PID controller.

If the line is under the left sensors, the error is positive → turn left.  
If it's under the right sensors, error is negative → turn right.

The rover doesn't know where it is. It doesn't know which way is north. It just follows the yellow brick road.

---

## The Sneaky Hard Part: Heading

We still need to know roughly which direction the rover is facing—not for navigation, but to correctly position those virtual sensors "ahead" of the rover.

No position data allowed, remember?

We estimate heading by **integrating the velocity commands**. If we told the left wheel to go faster than the right, the rover turned right. Integrate over time, and you get an orientation estimate.

But this drifts. Wheels slip. Small errors compound.

The fix: occasionally correct the heading using the guide rail itself. Use `cv2.fitLine` on the path ahead and blend it with the odometry estimate—98% odometry, 2% vision.

```python
heading = 0.98 * odometry_heading + 0.02 * visual_heading
```

Why 98/2? High odometry weight keeps motion smooth. Low vision weight kills drift without causing jitter.

---

## The Final State Machine

Two states.

1. **WALL_FOLLOW**: Generate guide rail. Follow it with PID. (99% of the time)
2. **LOOP_ESCAPE**: If stuck for 80+ frames, rotate in place to break free.

That's it. No turn states. No post-turn stabilization. No discrete orientation values.

---

## Results

**3 AM**: Full maze completion.  
**4 AM**: Submission uploaded.  
**1st Place** at IEEE Robotrix 2025-26.

---

## The Takeaway

The winning approach wasn't better at solving the original problem—it was solving a different problem entirely.

Instead of navigating a maze, we followed a line.  
Instead of tracking orientation, we regenerated perception every frame.  
Instead of hand-coded turn logic, we let PID do the work.

Sometimes the best optimization is to reframe the problem until it becomes trivial.

---

## Code

The full solution, approach documentation, and competition video are available on GitHub:

**🔗 [github.com/ADRSH99/Robotrix-26](https://github.com/ADRSH99/Robotrix-26)**

---

*Team Robodih | IEEE Robotrix 2025-26 | 🏆 1st Place*
