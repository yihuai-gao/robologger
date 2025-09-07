from robologger.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt
import zarr
import os
from typing import Dict, Any
from loguru import logger
import shutil

class CartesianCtrlLogger(BaseLogger):
    """Logger for cartesian control data.
    
    Naming Convention:
    - Must use RobotName enum values: right_arm, left_arm, head, body, left_end_effector, right_end_effector
    """
    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
    ):
        """Initialize cartesian control logger."""
        self._validate_robot_name(name)
        super().__init__(name, endpoint, attr)
        
        self.state_count = 0
        self.target_count = 0

    def _validate_robot_name(self, name: str) -> None:
        """Validate logger name matches RobotName enum pattern."""
        from robologger.utils.classes import RobotName
        
        valid_names = [robot.value for robot in RobotName]
        if name not in valid_names:
            raise ValueError(
                f"CartesianCtrlLogger name '{name}' must match RobotName enum.\n"
                f"Valid names: {valid_names}"
            )

    def _init_storage(self):
        """Initialize zarr storage for cartesian control data"""
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")
        
        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")

        try: 
            self.zarr_group = zarr.open_group(zarr_path, mode="w")
            assert self.zarr_group is not None, "Zarr group is not initialized"
            logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")
            
            self.zarr_group.create_dataset(
                "state_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float32, # NOTE: use float32 for timestamps is prolly enough
            )
            self.zarr_group.create_dataset(
                "target_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float32,
            )
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
        state_pos_xyz: npt.NDArray[np.float32],
        state_quat_wxyz: npt.NDArray[np.float32],
    ):
        """Log robot state (current pose)."""
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received state command")
            return
        
        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log state: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging states to make sure the zarr group is initialized.")
        
        try:
            # validate inputs
            if len(state_pos_xyz) != 3:
                raise ValueError(f"Expected 3D position, got {len(state_pos_xyz)}")
            if len(state_quat_wxyz) != 4:
                raise ValueError(f"Expected quaternion (w,x,y,z), got {len(state_quat_wxyz)}")
            
            for dataset_name in ["state_timestamps", "state_pos_xyz", "state_quat_wxyz"]:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"
                
                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)
                
            self.zarr_group["state_timestamps"][self.state_count] = state_timestamp
            self.zarr_group["state_pos_xyz"][self.state_count] = state_pos_xyz
            self.zarr_group["state_quat_wxyz"][self.state_count] = state_quat_wxyz
            
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
        target_pos_xyz: npt.NDArray[np.float32],
        target_quat_wxyz: npt.NDArray[np.float32],
    ):
        """Log robot target (desired pose)."""
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received target command")
            return
        
        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log target: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging targets to make sure the zarr group is initialized.")
        
        try:
            # validate inputs
            if len(target_pos_xyz) != 3:
                raise ValueError(f"Expected 3D position, got {len(target_pos_xyz)}")
            if len(target_quat_wxyz) != 4:
                raise ValueError(f"Expected quaternion (w,x,y,z), got {len(target_quat_wxyz)}")
            
            for dataset_name in ["target_timestamps", "target_pos_xyz", "target_quat_wxyz"]:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"

                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)
                
            self.zarr_group["target_timestamps"][self.target_count] = target_timestamp
            self.zarr_group["target_pos_xyz"][self.target_count] = target_pos_xyz
            self.zarr_group["target_quat_wxyz"][self.target_count] = target_quat_wxyz
            
            self.target_count += 1
            self.last_timestamp = target_timestamp

            if self.target_count % 1000 == 0:
                logger.debug(f"[{self.name}] Logged {self.target_count} targets")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log target: {e}")
            raise


    def get_stats(self) -> Dict[str, Any]:
          """Get logging statistics."""
          stats = {
              "episode_dir": self.episode_dir,
              "state_count": self.state_count,
              "target_count": self.target_count,
              "last_timestamp": self.last_timestamp,
              "storage_initialized": self.zarr_group is not None
          }
          logger.debug(f"[{self.name}] Current stats: {stats}")
          return stats
