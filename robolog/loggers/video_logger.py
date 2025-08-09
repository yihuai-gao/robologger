from robolog.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt

class VideoLogger(BaseLogger):
    def __init__(self, root_dir: str, project_name: str, task_name: str, run_name: str):
        super().__init__(root_dir, project_name, task_name, run_name)

    def _init_storage(self):
        ...
        # Init ffmpeg

    def _close_storage(self):
        ...
        # Close ffmpeg

    def log(
        self,
        *,
        timestamp: float,
        frame_dict: dict[str, npt.NDArray[np.uint8]],
    ):
        super().log(timestamp=timestamp)
        # TODO (jinyun): log video data

        # Check whether frame_dict has all the keys in self.attr

        
