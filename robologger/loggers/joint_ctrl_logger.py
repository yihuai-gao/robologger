from robologger.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt
import zarr
import os
from typing import Dict, Any
from loguru import logger
import shutil

class JointCtrlLogger(BaseLogger):
    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
    ):
        super().__init__(name, endpoint, attr)
        
        self.state_count = 0
        self.target_count = 0

    def _init_storage(self):
        """Initialize zarr storage for joint control data"""
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
            logger.info(f"[{self.name}] Created episode directory: {episode_dir}")
        else:
            logger.info(f"[{self.name}] Episode directory already exists: {episode_dir}")
            logger.info(f"[{self.name}] Deleting existing directory and creating a new one")
            shutil.rmtree(episode_dir)
            os.makedirs(episode_dir)

        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")

        try: 
            self.zarr_group = zarr.open_group(zarr_path, mode="w")
            assert self.zarr_group is not None, "Zarr group is not initialized"
            logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")
            
            num_joints = self.attr.get("num_joints")
            if num_joints is None:
                raise ValueError("num_joints must be specified in attr dictionary")
            
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
            self.zarr_group.create_dataset(
                "state_joint_pos",
                shape=(0, num_joints),
                chunks=(1000, num_joints),
                dtype=np.float32,
            )
            self.zarr_group.create_dataset(
                "target_joint_pos",
                shape=(0, num_joints),
                chunks=(1000, num_joints),
                dtype=np.float32,
            )

            self.state_count = 0
            self.target_count = 0

            logger.info(f"[{self.name}] Initialized zarr storage at {zarr_path} with {num_joints} joints")

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
        state_joint_pos: npt.NDArray[np.float32],
    ):
        """Log robot state (current joint positions)"""
        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log state: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging states to make sure the zarr group is initialized.")
        
        try:
            # Get expected number of joints from dataset shape
            expected_joints = self.zarr_group["state_joint_pos"].shape[1]
            
            # Validate inputs
            if len(state_joint_pos) != expected_joints:
                raise ValueError(f"Expected {expected_joints} joint positions, got {len(state_joint_pos)}")
            
            for dataset_name in ["state_timestamps", "state_joint_pos"]:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"
                
                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)
                
            self.zarr_group["state_timestamps"][self.state_count] = state_timestamp
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
        target_joint_pos: npt.NDArray[np.float32],
    ):
        """Log robot target (desired joint positions)"""
        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log target: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging targets to make sure the zarr group is initialized.")
        
        try:
            # Get expected number of joints from dataset shape
            expected_joints = self.zarr_group["target_joint_pos"].shape[1]
            
            # Validate inputs
            if len(target_joint_pos) != expected_joints:
                raise ValueError(f"Expected {expected_joints} joint positions, got {len(target_joint_pos)}")
            
            for dataset_name in ["target_timestamps", "target_joint_pos"]:
                assert dataset_name in self.zarr_group, f"Dataset {dataset_name} not found in zarr group"
                
                dataset = self.zarr_group[dataset_name]
                assert isinstance(dataset, zarr.Array), f"Dataset {dataset_name} must be a zarr.Array"

                original_shape = dataset.shape
                new_shape = (original_shape[0] + 1, *original_shape[1:])
                dataset.resize(new_shape)
                
            self.zarr_group["target_timestamps"][self.target_count] = target_timestamp
            self.zarr_group["target_joint_pos"][self.target_count] = target_joint_pos
            
            self.target_count += 1
            self.last_timestamp = target_timestamp

            if self.target_count % 1000 == 0:
                logger.debug(f"[{self.name}] Logged {self.target_count} targets")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log target: {e}")
            raise


    def get_stats(self) -> Dict[str, Any]:
          """Utility function to get logging statistics"""
          stats = {
              "episode_dir": self.episode_dir,
              "state_count": self.state_count,
              "target_count": self.target_count,
              "last_timestamp": self.last_timestamp,
              "storage_initialized": self.zarr_group is not None
          }
          logger.debug(f"[{self.name}] Current stats: {stats}")
          return stats
