"""
Robot arm control loop with robologger

Demonstrates how to integrate CtrlLogger into a robot control loop:
1. Initialize CtrlLogger with robot name and control configuration
2. Run control loop at desired frequency
3. Check recording state with update_recording_state()
4. Log both state (actual pose) and target (commanded pose) each iteration
5. State and target can have different timestamps

The logger stores pose data (position + quaternion) and timestamps in zarr format.
"""
import time
import numpy as np
from robologger.loggers.ctrl_logger import CtrlLogger

def main():
    # Initialize CtrlLogger for Cartesian-controlled arm
    logger = CtrlLogger(
        name="right_arm",                                  # Must match RobotName enum (see utils/classes.py)
                                                           # Options: right_arm, left_arm, head, body...
        endpoint="tcp://localhost:55556",                  # RMQ endpoint for main process
        attr={
            "robot_name": "right_arm",                     # Robot identifier (should match name)
            "ctrl_freq": 125.0,                            # Control loop frequency in Hz
            "num_joints": 7                                # Number of joints (optional if not logging joints)
        },
        log_eef_pose=True,                                 # Log end-effector pose
        log_joint_positions=False,                         # Don't log joint positions (Cartesian control only)
        target_type="eef_pose",                            # Control target is EEF pose
        joint_units=None                                   # Not needed since we're not logging joints
    )

    ctrl_freq = 125.0 # Control loop frequency in Hz
    dt = 1.0 / ctrl_freq

    # Randomly initialize robot state (actual pose from sensors/encoders)
    state_pos_xyz = np.array([0.3, 0.0, 0.5], dtype=np.float32)    # Position in meters (x, y, z)
    state_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Quaternion (w, x, y, z)

    # Randomly initialize target pose (commanded poses)
    target_pos_xyz = np.array([0.35, 0.05, 0.55], dtype=np.float32)
    target_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print(f"Robot control loop running at {ctrl_freq} Hz. Press Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.monotonic()

            # Get actual robot state (replace with real robot API)
            # e.g., state_pos_xyz, state_quat_wxyz = robot.get_eef_pose()
            state_pos_xyz += np.random.uniform(-0.001, 0.001, 3).astype(np.float32)
            state_timestamp = time.monotonic()

            # Get commanded target (replace with controller output)
            # e.g., target_pos_xyz, target_quat_wxyz = controller.get_target()
            target_pos_xyz += np.random.uniform(-0.002, 0.002, 3).astype(np.float32)
            target_timestamp = time.monotonic()

            # Check if main process has requested recording
            if logger.update_recording_state():
                # Log actual state (measured from robot)
                logger.log_state(
                    state_timestamp=state_timestamp,   # When state was measured
                    state_pos_xyz=state_pos_xyz,       # Actual position (3,) float32
                    state_quat_wxyz=state_quat_wxyz    # Actual orientation (4,) float32, w-first
                )
                # Log commanded target (from controller)
                logger.log_target(
                    target_timestamp=target_timestamp, # When target was commanded
                    target_pos_xyz=target_pos_xyz,     # Target position (3,) float32
                    target_quat_wxyz=target_quat_wxyz  # Target orientation (4,) float32, w-first
                )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()