"""
Joint-controlled arm with EEF pose logging

Demonstrates the most common robot control scenario:
1. Robot is controlled via joint positions (joint_pos target)
2. Robot can compute EEF pose via forward kinematics
3. Log BOTH joint positions and inferred EEF pose for downstream use

- Most robot arms (UR, Franka, Kinova, etc.)
- Learning policies that need both joint and Cartesian representations
- Debugging: compare commanded joints vs actual joints vs resulting EEF pose
"""
import time
import numpy as np
from robologger.loggers.ctrl_logger import CtrlLogger

def main():
    # Initialize CtrlLogger for joint-controlled arm with EEF pose logging
    logger = CtrlLogger(
        name="right_arm",                                  # Must match RobotName enum
        endpoint="tcp://localhost:55556",                  # RMQ endpoint for main process (same as robot_control_loop.py)
        attr={
            "robot_name": "right_arm",                     # Robot identifier (should match name)
            "ctrl_freq": 100.0,                            # Control loop frequency in Hz
            "num_joints": 7                                # Number of joints in the arm
        },
        log_eef_pose=True,                                 # Log EEF pose (from forward kinematics)
        log_joint_pos=True,                          # Log joint positions (primary control)
        target_type="joint_pos",                     # Control targets are joint positions
        joint_units="radians"                              # Joint positions in radians
    )

    ctrl_freq = 100.0
    dt = 1.0 / ctrl_freq

    # Initialize robot state (actual measurements from sensors)
    state_joint_pos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)  # 7 joints
    state_pos_xyz = np.array([0.3, 0.0, 0.5], dtype=np.float32)       # Inferred from FK
    state_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Initialize target joint positions (commanded by controller)
    target_joint_pos = np.array([0.1, -0.8, 0.05, -2.4, 0.0, 1.6, 0.8], dtype=np.float32)

    print(f"Joint-controlled arm loop running at {ctrl_freq} Hz. Press Ctrl+C to stop.")
    print("This example logs both joint positions (control) and EEF pose (inferred via FK)")

    try:
        while True:
            loop_start = time.monotonic()

            # Get actual robot state from sensors
            # In real robot: state_joint_pos = robot.get_joint_pos()
            state_joint_pos += np.random.uniform(-0.001, 0.001, 7).astype(np.float32)

            # Compute EEF pose via forward kinematics
            # In real robot: state_pos_xyz, state_quat_wxyz = robot.forward_kinematics(state_joint_pos)
            state_pos_xyz += np.random.uniform(-0.0005, 0.0005, 3).astype(np.float32)
            state_timestamp = time.monotonic()

            # Get commanded target joint positions from controller
            # In real robot: target_joint_pos = controller.get_target_joints()
            target_joint_pos += np.random.uniform(-0.002, 0.002, 7).astype(np.float32)
            target_timestamp = time.monotonic()

            # Check if main process has requested recording
            if logger.update_recording_state():
                # Log actual state (measured joints + inferred EEF pose)
                logger.log_state(
                    state_timestamp=state_timestamp,
                    state_joint_pos=state_joint_pos,       # Actual joint positions (7,) float32
                    state_pos_xyz=state_pos_xyz,           # EEF position from FK (3,) float32
                    state_quat_wxyz=state_quat_wxyz        # EEF orientation from FK (4,) float32
                )

                # Log commanded target (joint positions only, since that's what we control)
                logger.log_target(
                    target_timestamp=target_timestamp,
                    target_joint_pos=target_joint_pos      # Target joint positions (7,) float32
                )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()
