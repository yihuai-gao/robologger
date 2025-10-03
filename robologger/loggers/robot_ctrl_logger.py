from robologger.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt
import zarr
import os
from typing import Dict, Any, Optional, Literal
from loguru import logger


class RobotCtrlLogger(BaseLogger):
    """Logger for robot control data (joint and/or cartesian control).

    Usage:
        # EEF-controlled arm that can observe joints
        logger = RobotCtrlLogger(
            name="right_arm", endpoint="tcp://localhost:5555", attr={"num_joints": 7},
            log_eef_pose=True, log_joint_pos=True, target_type="eef_pose", joint_units="radians"
        )
        logger.log_state(state_timestamp=t, state_pos_xyz=pos, state_quat_wxyz=quat, state_joint_pos=joints)
        logger.log_target(target_timestamp=t, target_pos_xyz=target_pos, target_quat_wxyz=target_quat)

        # Joint-controlled arm that can infer EEF pose (common - log both)
        logger = RobotCtrlLogger(
            name="left_arm", endpoint="tcp://localhost:5557", attr={"num_joints": 7},
            log_eef_pose=True, log_joint_pos=True, target_type="joint_pos", joint_units="radians"
        )
        logger.log_state(state_timestamp=t, state_pos_xyz=pos, state_quat_wxyz=quat, state_joint_pos=joints)
        logger.log_target(target_timestamp=t, target_joint_pos=target_pos)

        # Joint-controlled gripper (meters, cannot infer EEF pose)
        logger = RobotCtrlLogger(
            name="right_end_effector", endpoint="tcp://localhost:5556", attr={"num_joints": 1},
            log_eef_pose=False, log_joint_pos=True, target_type="joint_pos", joint_units="meters"
        )
        logger.log_state(state_timestamp=t, state_joint_pos=joint_pos)
        logger.log_target(target_timestamp=t, target_joint_pos=target_pos)

    Configuration:
        log_eef_pose: Log end-effector pose (xyz + quaternion)
        log_joint_pos: Log joint positions
        target_type: Control target type ("eef_pose" or "joint_pos")
        joint_units: Joint units ("radians", "meters", or None if not logging joints)

    Notes:
        - Joint-controlled robots can often infer EEF pose via forward kinematics.
          Set log_eef_pose=True to log both joint positions and inferred EEF pose.
        - EEF-controlled robots may have joint encoders. Set log_joint_pos=True
          to log both EEF pose and observed joint positions.

    Constraints:
        - At least one of log_eef_pose or log_joint_pos must be True
        - target_type must match a logged observation type
        - joint_units required if log_joint_pos=True, None otherwise
        - attr["num_joints"] required if log_joint_pos=True

    Naming Convention:
        Must use RobotName enum: right_arm, left_arm, head, body, left_end_effector, right_end_effector
    """
    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
        log_eef_pose: bool,
        log_joint_pos: bool,
        target_type: Literal["eef_pose", "joint_pos"],
        joint_units: Optional[Literal["radians", "meters"]] = None,
    ):
        """Initialize control logger.

        Args:
            name: Logger name (must match RobotName enum)
            endpoint: Endpoint for data
            attr: Attributes dict (must contain 'num_joints' if log_joint_pos=True)
            log_eef_pose: Whether to log end-effector pose
            log_joint_pos: Whether to log joint positions
            target_type: Type of control target ("eef_pose" or "joint_pos")
            joint_units: Units for joint positions ("radians", "meters", or None)
        """
        self._validate_robot_name(name)

        # validate config
        if not log_eef_pose and not log_joint_pos:
            raise ValueError("At least one of log_eef_pose or log_joint_pos must be True")

        if target_type not in ["eef_pose", "joint_pos"]:
            raise ValueError(f"target_type must be 'eef_pose' or 'joint_pos', got '{target_type}'")

        if target_type == "eef_pose" and not log_eef_pose:
            raise ValueError("target_type='eef_pose' requires log_eef_pose=True")
        if target_type == "joint_pos" and not log_joint_pos:
            raise ValueError("target_type='joint_pos' requires log_joint_pos=True")

        if log_joint_pos:
            if joint_units not in ["radians", "meters"]:
                raise ValueError(f"joint_units must be 'radians' or 'meters' when log_joint_pos=True, got {joint_units}")
            if "num_joints" not in attr:
                raise ValueError("num_joints must be specified in attr when log_joint_pos=True")
        else:
            if joint_units is not None:
                raise ValueError(f"joint_units must be None when log_joint_pos=False, got {joint_units}")

        # store config
        self.log_eef_pose = log_eef_pose
        self.log_joint_pos = log_joint_pos
        self.target_type = target_type
        self.joint_units = joint_units
        self.num_joints = attr.get("num_joints") if log_joint_pos else None

        super().__init__(name, endpoint, attr)

        self.state_count = 0
        self.target_count = 0

    def _validate_robot_name(self, name: str) -> None:
        """Validate logger name matches RobotName enum pattern."""
        from robologger.utils.classes import RobotName

        valid_names = [robot.value for robot in RobotName]
        if name not in valid_names:
            raise ValueError(
                f"RobotCtrlLogger name '{name}' must match RobotName enum.\n"
                f"Valid names: {valid_names}"
            )

    def _init_storage(self):
        """Initialize zarr storage for control data"""
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")

        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")

        try:
            self.zarr_group = zarr.open_group(zarr_path, mode="w")
            assert self.zarr_group is not None, "Zarr group is not initialized"
            logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")

            logger.info(f"[{self.name}] Logger configuration: {self.get_stats}")

            # store metadata
            self.zarr_group.attrs.update({
                "target_type": self.target_type,
                "log_eef_pose": self.log_eef_pose,
                "log_joint_pos": self.log_joint_pos,
                "joint_units": self.joint_units,
                "num_joints": self.num_joints,
            })

            # timestamps
            self.zarr_group.create_dataset(
                "state_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float32,
            )
            self.zarr_group.create_dataset(
                "target_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float32,
            )

            # eef datasets
            if self.log_eef_pose:
                self.zarr_group.create_dataset(
                    "state_pos_xyz",
                    shape=(0, 3),
                    chunks=(1000, 3),
                    dtype=np.float32,
                )
                self.zarr_group.create_dataset(
                    "state_quat_wxyz",
                    shape=(0, 4),
                    chunks=(1000, 4),
                    dtype=np.float32,
                )

            # joint pos datasets
            if self.log_joint_pos:
                self.zarr_group.create_dataset(
                    "state_joint_pos",
                    shape=(0, self.num_joints),
                    chunks=(1000, self.num_joints),
                    dtype=np.float32,
                )

            if self.target_type == "joint_pos":
                self.zarr_group.create_dataset(
                    "target_joint_pos",
                    shape=(0, self.num_joints),
                    chunks=(1000, self.num_joints),
                    dtype=np.float32,
                )
            elif self.target_type == "eef_pose":
                self.zarr_group.create_dataset(
                    "target_pos_xyz",
                    shape=(0, 3),
                    chunks=(1000, 3),
                    dtype=np.float32,
                )
                self.zarr_group.create_dataset(
                    "target_quat_wxyz",
                    shape=(0, 4),
                    chunks=(1000, 4),
                    dtype=np.float32,
                )
            else:
                raise ValueError(f"Invalid target_type: {self.target_type}")

            self.state_count = 0
            self.target_count = 0

            logger.info(f"[{self.name}] Initialized zarr storage at {zarr_path}")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize zarr storage: {e}")
            raise


    def _close_storage(self):
        """Close zarr storage"""
        if self.zarr_group is not None:
            logger.info(f"[{self.name}] Closing zarr storage. Recorded {self.state_count} states and {self.target_count} targets")
            self.zarr_group = None
        else:
            logger.warning(f"[{self.name}] Attempted to close storage but zarr_group is None")

    def log_state(
        self,
        *,
        state_timestamp: float,
        state_pos_xyz: Optional[npt.NDArray[np.float32]] = None,
        state_quat_wxyz: Optional[npt.NDArray[np.float32]] = None,
        state_joint_pos: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Log robot state (current pose and/or joint positions).

        Args:
            state_timestamp: Timestamp of the state
            state_pos_xyz: EEF position (required if log_eef_pose=True)
            state_quat_wxyz: EEF orientation as quaternion (required if log_eef_pose=True)
            state_joint_pos: Joint positions (required if log_joint_pos=True)
        """
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received state command")
            return

        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log state: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging states to make sure the zarr group is initialized.")

        try:
            datasets_to_resize = ["state_timestamps"]

            if self.log_eef_pose:
                if state_pos_xyz is None or state_quat_wxyz is None:
                    raise ValueError("state_pos_xyz and state_quat_wxyz are required when log_eef_pose=True")

                if len(state_pos_xyz) != 3:
                    raise ValueError(f"Expected 3D position, got {len(state_pos_xyz)}")
                if len(state_quat_wxyz) != 4:
                    raise ValueError(f"Expected quaternion (w,x,y,z), got {len(state_quat_wxyz)}")

                datasets_to_resize.extend(["state_pos_xyz", "state_quat_wxyz"])

            if self.log_joint_pos:
                if state_joint_pos is None:
                    raise ValueError("state_joint_pos is required when log_joint_pos=True")

                if len(state_joint_pos) != self.num_joints:
                    raise ValueError(f"Expected {self.num_joints} joint positions, got {len(state_joint_pos)}")

                datasets_to_resize.append("state_joint_pos")

            for dataset_name in datasets_to_resize:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"

                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)

            self.zarr_group["state_timestamps"][self.state_count] = state_timestamp

            if self.log_eef_pose:
                self.zarr_group["state_pos_xyz"][self.state_count] = state_pos_xyz
                self.zarr_group["state_quat_wxyz"][self.state_count] = state_quat_wxyz

            if self.log_joint_pos:
                self.zarr_group["state_joint_pos"][self.state_count] = state_joint_pos

            self.state_count += 1
            self.last_timestamp = state_timestamp

            if self.state_count % 1000 == 0:
                logger.debug(f"[{self.name}] Logged {self.state_count} states")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log state: {e}")
            raise

    def log_target(
        self,
        *,
        target_timestamp: float,
        target_pos_xyz: Optional[npt.NDArray[np.float32]] = None,
        target_quat_wxyz: Optional[npt.NDArray[np.float32]] = None,
        target_joint_pos: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Log robot target (desired pose and/or joint positions).

        Args:
            target_timestamp: Timestamp of the target
            target_pos_xyz: Target EEF position (required if target_type="eef_pose")
            target_quat_wxyz: Target EEF orientation (required if target_type="eef_pose")
            target_joint_pos: Target joint positions (required if target_type="joint_pos")
        """
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received target command")
            return

        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log target: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging targets to make sure the zarr group is initialized.")

        try:
            datasets_to_resize = ["target_timestamps"]

            if self.target_type == "eef_pose":
                if target_pos_xyz is None or target_quat_wxyz is None:
                    raise ValueError("target_pos_xyz and target_quat_wxyz are required when target_type='eef_pose'")

                if len(target_pos_xyz) != 3:
                    raise ValueError(f"Expected 3D position, got {len(target_pos_xyz)}")
                if len(target_quat_wxyz) != 4:
                    raise ValueError(f"Expected quaternion (w,x,y,z), got {len(target_quat_wxyz)}")

                datasets_to_resize.extend(["target_pos_xyz", "target_quat_wxyz"])

            elif self.target_type == "joint_pos":
                if target_joint_pos is None:
                    raise ValueError("target_joint_pos is required when target_type='joint_pos'")

                if len(target_joint_pos) != self.num_joints:
                    raise ValueError(f"Expected {self.num_joints} joint positions, got {len(target_joint_pos)}")

                datasets_to_resize.append("target_joint_pos")

            for dataset_name in datasets_to_resize:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"

                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)

            self.zarr_group["target_timestamps"][self.target_count] = target_timestamp

            if self.target_type == "eef_pose":
                self.zarr_group["target_pos_xyz"][self.target_count] = target_pos_xyz
                self.zarr_group["target_quat_wxyz"][self.target_count] = target_quat_wxyz

            elif self.target_type == "joint_pos":
                self.zarr_group["target_joint_pos"][self.target_count] = target_joint_pos

            self.target_count += 1
            self.last_timestamp = target_timestamp

            if self.target_count % 1000 == 0:
                logger.debug(f"[{self.name}] Logged {self.target_count} targets")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log target: {e}")
            raise

    @property
    def get_stats(self) -> Dict[str, Any]:
          """Get logging statistics."""
          stats = {
              "episode_dir": self.episode_dir,
              "state_count": self.state_count,
              "target_count": self.target_count,
              "last_timestamp": self.last_timestamp,
              "storage_initialized": self.zarr_group is not None,
              "log_eef_pose": self.log_eef_pose,
              "log_joint_pos": self.log_joint_pos,
              "target_type": self.target_type,
              "joint_units": self.joint_units,
          }
          logger.debug(f"[{self.name}] Current stats: {stats}")
          return stats
