from typing import Dict
from robolog.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt

class VideoLogger(BaseLogger):
    def __init__(self, name: str, root_dir: str, project_name: str, task_name: str, run_name: str, attr: dict):
        super().__init__(name, root_dir, project_name, task_name, run_name, attr)

    def _init_storage(self):
        ...
        # Init ffmpeg

    def _close_storage(self):
        ...
        # Close ffmpeg

    def log_frame(
        self,
        *,
        camera_name: str,
        timestamp: float,
        frame: npt.NDArray[np.uint8],
    ):
        # TODO (jinyun): log video data

        # Check whether frame_dict has all the keys in self.attr
        for key in self.attr["camera_config"].keys():
            if key not in frame_dict:
                raise ValueError(f"Frame dictionary does not have key: {key}")
            if self.attr["camera_config"][key]["type"] == "rgb":
                frame_rgb_hwc = ...
            elif self.attr["camera_config"][key]["type"] == "depth":
                # TODO: use log_scale + hue_encoding
                frame_rgb_hwc = ...

            # TODO: write frame to ffmpeg




        
