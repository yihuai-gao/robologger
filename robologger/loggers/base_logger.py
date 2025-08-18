from abc import ABC, abstractmethod
import os
import sys
from typing import Any, Dict, Optional
from loguru import logger
import zarr
from enum import Enum
import robotmq
from robotmq import RMQServer, deserialize, serialize
from robologger.utils.stdout_setup import setup_logging

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
        endpoint: str, # "tcp://0.0.0.0:55555"
        attr: Dict[str, Any],
    ):
        setup_logging()
        
        self.name: str = name
        self.last_timestamp: float = 0.0
        self.attr: Dict[str, Any] = attr
        self.zarr_group: Optional[zarr.Group] = None

        self.rmq_server = RMQServer(server_name=name, server_endpoint=endpoint)
        self.rmq_server.add_topic(topic="command", message_remaining_time_s=10.0)
        self.rmq_server.add_topic(topic="info", message_remaining_time_s=10.0)

        self._is_recording: bool = False
        self.rmq_server.put_data(topic="info", data=serialize({"name": self.name, "attr": self.attr}))

    @property
    def is_recording(self) -> bool:
        raw_data, timestamps = self.rmq_server.pop_data(topic="command", n=0) # Clear the entire queue
        if len(raw_data) > 0:
            # Only consider the last message
            command = deserialize(raw_data[-1])
            assert isinstance(command, dict), "Command must be a dictionary"
            if command["type"] == "start":
                if self._is_recording:
                    logger.error("Already recording, but received start command. Stopping the current recording.")
                    self.stop_recording()
                self.start_recording(command["episode_dir"])
            elif command["type"] == "stop":
                if not self._is_recording:
                    raise RuntimeError("Not recording, but received stop command")
                self.stop_recording()
            else:
                raise ValueError(f"Unknown command type: {command['type']}, must be 'start' or 'end'")
            
        return self._is_recording

    def start_recording(self, episode_dir: str):
        self._is_recording = True

        if episode_idx is not None:
            self.episode_idx = episode_idx
        else:
            self.episode_idx = self._get_next_episode_idx()
        assert self.episode_idx >= 0, "Episode index must be non-negative"
        logger.info(f"Starting episode {self.episode_idx}")

        self._init_storage()
        self._set_attributes()

    def stop_recording(self):
        self._is_recording = False
        logger.info(f"Ending episode {self.episode_idx}")
        self._close_storage()
        self.episode_idx = -1

    def _set_attributes(self):
        """Set attributes on the zarr group"""
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Setting attributes on the zarr group: {self.attr}")
        for key, value in self.attr.items():
            self.zarr_group.attrs[key] = value

    @abstractmethod
    def _init_storage(self):
        ...

    @abstractmethod
    def _close_storage(self):
        ...
