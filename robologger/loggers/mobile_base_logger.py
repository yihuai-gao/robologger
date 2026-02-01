from robologger.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt
import zarr
import os
from typing import Dict, Any, Optional
from loguru import logger


class MobileBaseLogger(BaseLogger):

    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
        log_state_pose: bool = False,
        log_state_velocity: bool = True,
        log_target_pose: bool = False,
        log_target_velocity: bool = True,
    ):
        if not log_state_pose and not log_state_velocity:
            raise ValueError("At least one of log_state_pose or log_state_velocity must be True")
        if not log_target_pose and not log_target_velocity:
            raise ValueError("At least one of log_target_pose or log_target_velocity must be True")

        self.log_state_pose = log_state_pose
        self.log_state_velocity = log_state_velocity
        self.log_target_pose = log_target_pose
        self.log_target_velocity = log_target_velocity

        super().__init__(name, endpoint, attr)
        self.state_count = 0
        self.target_count = 0

    def _init_storage(self):
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")

        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")

        try:
            self.zarr_group = zarr.open_group(zarr_path, mode="w")
            assert self.zarr_group is not None, "Zarr group is not initialized"
            logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")
            logger.info(f"[{self.name}] Logger configuration: {self.get_stats}")

            self.zarr_group.attrs.update({
                "log_state_pose": self.log_state_pose,
                "log_state_velocity": self.log_state_velocity,
                "log_target_pose": self.log_target_pose,
                "log_target_velocity": self.log_target_velocity,
            })

            self.data_lists["state_timestamps"] = []
            self.data_lists["target_timestamps"] = []

            if self.log_state_pose:
                self.data_lists["state_pose"] = []
            if self.log_state_velocity:
                self.data_lists["state_velocity"] = []
            if self.log_target_pose:
                self.data_lists["target_pose"] = []
            if self.log_target_velocity:
                self.data_lists["target_velocity"] = []

            self.state_count = 0
            self.target_count = 0

            logger.info(f"[{self.name}] Initialized zarr storage at {zarr_path}")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize zarr storage: {e}")
            raise

    def _close_storage(self):
        if self.zarr_group is not None:
            logger.info(f"[{self.name}] Closing zarr storage. Recorded {self.state_count} states and {self.target_count} targets")
            self.zarr_group = None
        else:
            logger.warning(f"[{self.name}] Attempted to close storage but zarr_group is None")

    def log_state(
        self,
        *,
        state_timestamp: float,
        state_pose: Optional[npt.NDArray[np.float64]] = None,
        state_velocity: Optional[npt.NDArray[np.float64]] = None,
    ):
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received state command")
            return

        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log state: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging states to make sure the zarr group is initialized.")

        try:
            if self.log_state_pose:
                if state_pose is None:
                    raise ValueError("state_pose is required when log_state_pose=True")
                if len(state_pose) != 3:
                    raise ValueError(f"Expected 3D pose (pos_x, pos_y, rot_yaw), got {len(state_pose)}")
                self.data_lists["state_pose"].append(state_pose.copy())

            if self.log_state_velocity:
                if state_velocity is None:
                    raise ValueError("state_velocity is required when log_state_velocity=True")
                if len(state_velocity) != 3:
                    raise ValueError(f"Expected 3D velocity (vel_x, vel_y, vel_yaw), got {len(state_velocity)}")
                self.data_lists["state_velocity"].append(state_velocity.copy())

            self.data_lists["state_timestamps"].append(state_timestamp)

            self.state_count += 1
            self.last_timestamp = state_timestamp

            if self.state_count % 1000 == 0:
                logger.debug(f"[{self.name}] Buffered {self.state_count} states in memory")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log state: {e}")
            raise

    def log_target(
        self,
        *,
        target_timestamp: float,
        target_pose: Optional[npt.NDArray[np.float64]] = None,
        target_velocity: Optional[npt.NDArray[np.float64]] = None,
    ):
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received target command")
            return

        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log target: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging targets to make sure the zarr group is initialized.")

        try:
            if self.log_target_pose:
                if target_pose is None:
                    raise ValueError("target_pose is required when log_target_pose=True")
                if len(target_pose) != 3:
                    raise ValueError(f"Expected 3D pose (pos_x, pos_y, rot_yaw), got {len(target_pose)}")
                self.data_lists["target_pose"].append(target_pose.copy())

            if self.log_target_velocity:
                if target_velocity is None:
                    raise ValueError("target_velocity is required when log_target_velocity=True")
                if len(target_velocity) != 3:
                    raise ValueError(f"Expected 3D velocity (vel_x, vel_y, vel_yaw), got {len(target_velocity)}")
                self.data_lists["target_velocity"].append(target_velocity.copy())

            self.data_lists["target_timestamps"].append(target_timestamp)

            self.target_count += 1
            self.last_timestamp = target_timestamp

            if self.target_count % 1000 == 0:
                logger.debug(f"[{self.name}] Buffered {self.target_count} targets in memory")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to log target: {e}")
            raise

    @property
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "episode_dir": self.episode_dir,
            "state_count": self.state_count,
            "target_count": self.target_count,
            "last_timestamp": self.last_timestamp,
            "storage_initialized": self.zarr_group is not None,
            "log_state_pose": self.log_state_pose,
            "log_state_velocity": self.log_state_velocity,
            "log_target_pose": self.log_target_pose,
            "log_target_velocity": self.log_target_velocity,
        }
        logger.debug(f"[{self.name}] Current stats: {stats}")
        return stats
