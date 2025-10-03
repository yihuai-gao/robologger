"""
Gripper control loop with robologger

Demonstrates how to integrate RobotCtrlLogger into a gripper control loop:
1. Initialize RobotCtrlLogger with end effector name and number of joints
2. Run control loop at desired frequency
3. Check recording state with update_recording_state()
4. Log both state (actual joint positions) and target (commanded positions)
5. Joint positions can be gripper width or any multi-finger joint values

The logger stores joint position data and timestamps in zarr format.
Works for any joint-based end effector (grippers, dex hands, etc.).
"""
import time
import numpy as np
from robologger.loggers.robot_ctrl_logger import RobotCtrlLogger

def main():
    # Initialize RobotCtrlLogger for joint-controlled gripper
    logger = RobotCtrlLogger(
        name="right_end_effector",                         # Must match RobotName enum (see utils/classes.py)
                                                           # Options: right_end_effector, left_end_effector
        endpoint="tcp://localhost:55557",                  # RMQ endpoint for main process
        attr={
            "robot_name": "right_end_effector",            # End effector identifier (should match name)
            "ctrl_freq": 30.0,                             # Control loop frequency in Hz
            "num_joints": 1                                # Number of joints (1 for parallel gripper, more for hands)
        },
        log_eef_pose=False,                                # Don't log end-effector pose (gripper has no EEF pose)
        log_joint_pos=True,                                # Log joint positions (gripper width)
        target_type="joint_pos",                           # Control target is joint positions
        joint_units="meters"                               # Joint units (meters for gripper width)
    )

    ctrl_freq = 30.0  
    dt = 1.0 / ctrl_freq

    # Randomly initialized gripper state (actual joint positions from sensors)
    # For WSG50: width in meters (0.0-0.1m range)
    state_joint_pos = np.array([0.08])     # 80mm open
    target_joint_pos = np.array([0.06])    # 60mm target

    print(f"Gripper control loop running at {ctrl_freq} Hz. Press Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.monotonic()

            # Get actual gripper state (replace with real gripper API)
            # e.g., state_joint_pos = gripper.get_width()  # for parallel gripper
            # e.g., state_joint_pos = hand.get_joint_pos()  # for multi-finger hand
            state_joint_pos += np.random.uniform(-0.001, 0.001, 1)
            state_joint_pos = np.clip(state_joint_pos, 0.0, 0.1)  # WSG50: 0-100mm range
            state_timestamp = time.monotonic()

            # Get commanded target (replace with controller output)
            # e.g., target_joint_pos = controller.get_target_width()
            target_joint_pos += np.random.uniform(-0.002, 0.002, 1)
            target_joint_pos = np.clip(target_joint_pos, 0.0, 0.11)
            target_timestamp = time.monotonic()

            # Check if main process has requested recording
            if logger.update_recording_state():
                # Log actual state (measured from gripper sensors)
                logger.log_state(
                    state_timestamp=state_timestamp,           # When state was measured
                    state_joint_pos=state_joint_pos.astype(np.float32)  # Actual joint positions (N,) float32
                )
                # Log commanded target (from controller)
                logger.log_target(
                    target_timestamp=target_timestamp,         # When target was commanded
                    target_joint_pos=target_joint_pos.astype(np.float32)  # Target joint positions (N,) float32
                )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()