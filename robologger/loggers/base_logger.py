from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Optional
from loguru import logger
from enum import Enum
import zarr


from robotmq import RMQServer, deserialize, serialize
from robologger.utils.stdout_setup import setup_logging
from atexit import register

class BaseLogger(ABC):
    """Abstract base class for all loggers with RMQ communication.

    Uses in-memory buffering during recording with batch zarr write at stop_recording().
    Supports pause/resume and remote control via RMQ command/info topics.
    """
    def __init__(
        self,
        name: str,
        endpoint: str, # "tcp://0.0.0.0:55555"
        attr: Dict[str, Any],
    ):
        """Initialize base logger with RMQ communication."""
        setup_logging()
        
        self.name: str = name
        self.last_timestamp: float = 0.0
        self.attr: Dict[str, Any] = attr
        self.zarr_group: Optional[zarr.Group] = None
        self.data_lists: Dict[str, List[npt.NDArray[np.float64]]] = {}

        self.rmq_server = RMQServer(server_name=name, server_endpoint=endpoint)
        self.rmq_server.add_topic(topic="command", message_remaining_time_s=10.0)
        self.rmq_server.add_topic(topic="info", message_remaining_time_s=10.0)

        self._is_recording: bool = False
        self.episode_dir: Optional[str] = None
        self.rmq_server.put_data(topic="info", data=serialize({"name": self.name, "attr": self.attr}))

        register(self.on_exit)

    def on_exit(self):
        if self._is_recording:
            self.stop_recording()

    def update_recording_state(self) -> bool:
        """Process recording commands from RMQ and update state."""
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
                    logger.error("Not recording, but received stop command")
                self.stop_recording()
            elif command["type"] == "pause":
                if not self._is_recording:
                    logger.error("Not recording, but received pause command")
                    return self._is_recording
                logger.info("Pausing recording")
                self._is_recording = False
            elif command["type"] == "resume":
                if self._is_recording:
                    logger.error("Already recording, but received resume command")
                    return self._is_recording
                if self.episode_dir is None or not os.path.exists(self.episode_dir):
                    logger.error("No episode directory set or directory doesn't exist, cannot resume recording")
                    return self._is_recording
                logger.info("Resuming recording")
                self._is_recording = True
            else:
                raise ValueError(f"Unknown command type received in update_recording_state() method: {command['type']}, must be 'start', 'stop', 'pause', or 'resume'")
            
        return self._is_recording

    def start_recording(self, episode_dir: str):
        """Start recording session and initialize storage."""
        self._is_recording = True
        self.episode_dir = episode_dir
        logger.info(f"Starting recording in episode directory: {self.episode_dir}")

        self._init_storage()
        self._set_attributes()

    def stop_recording(self) -> bool:
        """Stop recording session and close storage.

        Returns:
            bool: True if all operations succeeded, False if dump failed (but cleanup still completes)
        """
        self._is_recording = False
        logger.info(f"Stopping recording: {self.episode_dir}")

        # Dump buffered data before closing storage
        dump_success = self._dump_data_to_zarr()
        if not dump_success:
            logger.error(f"[{self.name}] Data dump failed, but continuing with cleanup")

        # Always complete cleanup even if dump failed
        self.episode_dir = None
        self._close_storage()

        return dump_success

    def _set_attributes(self):
        """Set attributes on the zarr group."""
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Setting attributes on the zarr group: {self.attr}")
        for key, value in self.attr.items():
            self.zarr_group.attrs[key] = value

    def _dump_data_to_zarr(self) -> bool:
        """Dump buffered data from data_lists to zarr storage.

        This is called automatically before closing storage to ensure all buffered
        data is safely written to disk in a single batch operation per dataset.

        Returns:
            bool: True if dump succeeded (or no data to dump), False if failed

        Note:
            This method catches all exceptions and returns False to allow graceful
            cleanup in stop_recording(). No exceptions are raised.
        """
        if self.zarr_group is None:
            logger.warning(f"[{self.name}] stop_recording() - Cannot dump data: zarr_group is None")
            return False

        if not self.data_lists:
            logger.debug(f"[{self.name}] stop_recording() - No data_lists to dump")
            return True 

        try:
            logger.info(f"[{self.name}] Starting data dump to zarr...")
            num_fields_written = 0
            total_samples = 0

            for datafield_name, data_list in self.data_lists.items():
                if not data_list:
                    logger.warning(f"[{self.name}] stop_recording() - Data field has no data: {datafield_name}")
                    continue

                try:
                    data_array = np.array(data_list, dtype=np.float64)

                    if len(data_array.shape) == 1:
                        # 1D array (timestamps)
                        shape = (len(data_array),)
                        chunks = (min(1000, len(data_array)),)
                    else:
                        # 2D array (positions, quaternions, joints)
                        shape = data_array.shape
                        chunks = (min(1000, shape[0]), shape[1])

                    self.zarr_group.create_dataset(
                        datafield_name,
                        data=data_array,
                        shape=shape,
                        chunks=chunks,
                        dtype=np.float64, # no compressor
                        overwrite=True,
                    )

                    num_fields_written += 1
                    total_samples += len(data_list)
                    logger.debug(f"[{self.name}] Dumped {datafield_name}: {len(data_list)} samples, shape={shape}")

                except Exception as datafield_error:  # pylint: disable=broad-except
                    logger.error(f"[{self.name}] stop_recording() - Failed to dump data field {datafield_name}: {datafield_error}")
                    continue

            if num_fields_written > 0:
                logger.info(f"[{self.name}] Data dump complete: {num_fields_written} data fields, {total_samples} total samples")
                return True
            else:
                logger.error(f"[{self.name}] stop_recording() - Data dump failed: no data fields were successfully written")
                return False

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"[{self.name}] stop_recording() - Error during data dump: {e}")
            return False  

    @abstractmethod
    def _init_storage(self):
        ...

    @abstractmethod
    def _close_storage(self):
        ...
