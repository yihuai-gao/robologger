from robologger.loggers.base_logger import BaseLogger
from typing import Dict, Any, Tuple
from loguru import logger
import os
import zarr
import numpy as np
import numpy.typing as npt

class GenericLogger(BaseLogger):

    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
        data_shapes: Dict[str, Tuple[int, ...]],
    ):
        super().__init__(name, endpoint, attr)
        self.data_shapes = data_shapes

    def _init_storage(self):

        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")

        for data_name, data_shape in self.data_shapes.items():
            self.data_lists[data_name] = []
        self.data_lists["timestamps"] = []

        # Only create a zarr group and save the attributes
        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")
        self.zarr_group = zarr.open_group(zarr_path, mode="w")

        self.zarr_group.attrs.update(self.attr)
        
    def log_data(self, *, timestamp: float, data_dict: Dict[str, npt.NDArray[np.float64]]):
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received data command")
            return

        if self.zarr_group is None:
            logger.warning(f"[{self.name}] Cannot log data: storage not initialized")
            raise ValueError("Storage not initialized. Please call start_episode() before logging data to make sure the zarr group is initialized.")

        for data_name, data_shape in self.data_shapes.items():
            if data_name not in data_dict:
                raise ValueError(f"Data name {data_name} not sent to the function")

            data_array = data_dict[data_name]
            if data_array.shape != data_shape:
                raise ValueError(f"Input data shape {data_array.shape} does not match expected shape {data_shape}")

            self.data_lists[data_name].append(data_array)

        self.data_lists["timestamps"].append(timestamp)