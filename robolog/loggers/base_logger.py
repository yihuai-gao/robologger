from abc import ABC, abstractmethod
import os
import sys
from typing import Any, Dict, Optional
from loguru import logger
import zarr
from enum import Enum

# global flag to ensure stdout logger only setup once
_logging_configured = False

def setup_logging(level: str = "INFO", format_str: Optional[str] = None, colorize: bool = True):
    """Setup custom loguru logging configuration."""
    global _logging_configured
    if _logging_configured:
        return
    
    if format_str is None:
        format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    logger.remove()  # remove default handler and add custom logger
    logger.add(sys.stdout, format=format_str, colorize=colorize, level=level)
    _logging_configured = True

class LoggerType(Enum):
    VIDEO = "video"
    JOINT_CTRL = "joint_ctrl"
    CARTESIAN_CTRL = "cartesian_ctrl"
    SENSOR = "sensor"
    GENERIC = "generic"

class BaseLogger(ABC):
    def __init__(
        self,
        name: str,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        attr: Dict[str, Any],
    ):
        self.name: str = name
        self.root_dir: str = root_dir

        setup_logging() # stdout logging

        if not os.path.exists(self.root_dir):
            logger.info(f"Creating root directory: {self.root_dir}")
            os.makedirs(self.root_dir)

        self.project_name: str = project_name
        self.task_name: str = task_name
        self.run_name: str = run_name
        self.last_timestamp: float = 0.0
        self.attr: dict = attr

        self.run_dir: str = os.path.join(self.root_dir, self.project_name, self.task_name, self.run_name)
        if not os.path.exists(self.run_dir):
            logger.info(f"Creating run directory: {self.run_dir}")
            os.makedirs(self.run_dir)

        self.episode_idx: int = -1
        self.zarr_group: Optional[zarr.Group] = None

    def start_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is not None:
            self.episode_idx = episode_idx
        else:
            self.episode_idx = self._get_next_episode_idx()
        logger.info(f"Starting episode {self.episode_idx}")

        self._init_storage()
        self._set_attributes()

    def end_episode(self):
        logger.info(f"Ending episode {self.episode_idx}")
        self._close_storage()
        self.episode_idx = -1

    def _set_attributes(self):
        """Set attributes on the zarr group"""
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Setting attributes on the zarr group: {self.attr}")
        for key, value in self.attr.items():
            self.zarr_group.attrs[key] = value
                
    def _get_next_episode_idx(self) -> int:
        """Find the next available episode index"""
        if not os.path.exists(self.run_dir):
            return 0
        
        existing_episodes = []
        for item in os.listdir(self.run_dir):
            if item.startswith("episode_") and os.path.isdir(os.path.join(self.run_dir, item)):
                try:
                    episode_num = int(item.split("_")[1])
                    existing_episodes.append(episode_num)
                except (IndexError, ValueError):
                    continue
        
        return max(existing_episodes, default=-1) + 1

    @abstractmethod
    def _init_storage(self):
        ...

    @abstractmethod
    def _close_storage(self):
        ...
