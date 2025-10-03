"""
Robot arm control loop with robologger

Demonstrates how to integrate RobotCtrlLogger into a robot control loop with two modes:

CARTESIAN MODE (--mode cartesian):
- End-effector pose control (minimal logging)
- Logs EEF pose state and target only
- Use for Cartesian/task-space controllers

JOINT MODE (--mode joint, RECOMMENDED):
- Joint position control (full data logging)
- Logs both joint positions AND inferred EEF pose
- Most common for robot arms (UR, Franka, Kinova, etc.)
- Better for learning policies and debugging

Usage:
    python examples/robot_control_loop.py --mode joint      # Joint control (RECOMMENDED)
    python examples/robot_control_loop.py --mode cartesian  # Cartesian/EEF control

Common workflow:
1. Initialize RobotCtrlLogger with robot name and control configuration
2. Run control loop at desired frequency
3. Check recording state with update_recording_state()
4. Log both state (actual pose) and target (commanded pose) each iteration
5. State and target can have different timestamps

The logger stores pose data (position + quaternion) and/or joint data in zarr format.
"""
import time
import argparse
import numpy as np
from robologger.loggers.robot_ctrl_logger import RobotCtrlLogger


def run_cartesian_mode():
    """
    Cartesian/EEF pose control mode - logs only EEF pose

    Demonstrates:
    1. Initialize RobotCtrlLogger for Cartesian-controlled arm
    2. Run control loop at desired frequency
    3. Check recording state with update_recording_state()
    4. Log both state (actual pose) and target (commanded pose) each iteration
    5. State and target can have different timestamps
    """
    # Initialize RobotCtrlLogger for Cartesian-controlled arm
    logger = RobotCtrlLogger(
        name="right_arm",                                  # Must match RobotName enum (see utils/classes.py)
                                                           # Options: right_arm, left_arm, head, body...
        endpoint="tcp://localhost:55556",                  # RMQ endpoint for main process
        attr={
            "robot_name": "right_arm",                     # Robot identifier (should match name)
            "ctrl_freq": 125.0,                            # Control loop frequency in Hz
            "num_joints": 7                                # Number of joints (optional if not logging joints)
        },
        log_eef_pose=True,                                 # Log end-effector pose
        log_joint_pos=False,                               # Don't log joint positions (Cartesian control only)
        target_type="eef_pose",                            # Control target is EEF pose
        joint_units=None                                   # Not needed since we're not logging joints
    )

    ctrl_freq = 125.0  # Control loop frequency in Hz
    dt = 1.0 / ctrl_freq

    # Randomly initialize robot state (actual pose from sensors/encoders)
    state_pos_xyz = np.array([0.3, 0.0, 0.5], dtype=np.float32)    # Position in meters (x, y, z)
    state_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Quaternion (w, x, y, z)

    # Randomly initialize target pose (commanded poses)
    target_pos_xyz = np.array([0.35, 0.05, 0.55], dtype=np.float32)
    target_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print(f"[CARTESIAN MODE] Robot control loop running at {ctrl_freq} Hz. Press Ctrl+C to stop.")
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


def run_joint_mode():
    """
    Joint-controlled arm with EEF pose logging (RECOMMENDED)

    Demonstrates the most common robot control scenario:
    1. Robot is controlled via joint positions (joint_pos target)
    2. Robot can compute EEF pose via forward kinematics
    3. Log BOTH joint positions and inferred EEF pose for downstream use

    Use cases:
    - Most robot arms (UR, Franka, Kinova, etc.)
    - Learning policies that need both joint and Cartesian representations
    - Debugging: compare commanded joints vs actual joints vs resulting EEF pose
    """
    # Initialize RobotCtrlLogger for joint-controlled arm with EEF pose logging
    logger = RobotCtrlLogger(
        name="right_arm",                                  # Must match RobotName enum
        endpoint="tcp://localhost:55556",                  # RMQ endpoint for main process
        attr={
            "robot_name": "right_arm",                     # Robot identifier (should match name)
            "ctrl_freq": 100.0,                            # Control loop frequency in Hz
            "num_joints": 7                                # Number of joints in the arm
        },
        log_eef_pose=True,                                 # Log EEF pose (from forward kinematics)
        log_joint_pos=True,                                # Log joint positions (primary control)
        target_type="joint_pos",                           # Control targets are joint positions
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

    print(f"[JOINT MODE] Joint-controlled arm loop running at {ctrl_freq} Hz. Press Ctrl+C to stop.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Robot control loop with robologger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Control Modes:
  joint      - Joint position control (RECOMMENDED - full data logging)
               Logs both joint positions and EEF pose
  cartesian  - Cartesian/EEF pose control (minimal logging)
               Logs only EEF pose state and target
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["joint", "cartesian"],
        default="joint",
        help="Control mode: 'joint' (recommended) or 'cartesian' (default: joint)"
    )

    args = parser.parse_args()

    if args.mode == "joint":
        run_joint_mode()
    elif args.mode == "cartesian":
        run_cartesian_mode()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
