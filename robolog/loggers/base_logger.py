from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional
from loguru import Logger
import zarr
from enum import Enum

class LoggerType(Enum):
    VIDEO = "video"
    JOINT_CTRL = "joint_ctrl"
    CARTESIAN_CTRL = "cartesian_ctrl"
    SENSOR = "sensor"
    GENERIC = "generic"

class BaseLogger(ABC):
    def __init__(self, name: str, root_dir: str, project_name: str, task_name: str, run_name: str, attr: Dict[str, Any]):
        self.name: str = name
        self.root_dir: str = root_dir
        self.loguru_logger: Logger = Logger()

        if not os.path.exists(self.root_dir):
            self.loguru_logger.info(f"Creating root directory: {self.root_dir}")
            os.makedirs(self.root_dir)

        self.project_name: str = project_name
        self.task_name: str = task_name
        self.run_name: str = run_name
        self.last_timestamp: float = 0.0
        self.attr: dict = attr

        self.run_dir: str = os.path.join(self.root_dir, self.project_name, self.task_name, self.run_name)
        if not os.path.exists(self.run_dir):
            self.loguru_logger.info(f"Creating run directory: {self.run_dir}")
            os.makedirs(self.run_dir)

        self.episode_idx: int = -1
        self.zarr_group: Optional[zarr.Group] = None

    def start_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is not None:
            self.episode_idx = episode_idx
        else:
            # TODO (jinyun): find the last episode index
            ...
        self.loguru_logger.info(f"Starting episode {self.episode_idx}")

        # TODO (jinyun): initialize storage
        self._init_storage()
        # TODO: set attr

    def end_episode(self):
        self.loguru_logger.info(f"Ending episode {self.episode_idx}")
        self._close_storage()
        self.episode_idx = -1


    @abstractmethod
    def _init_storage(self):
        ...

    @abstractmethod
    def _close_storage(self):
        ...



    
