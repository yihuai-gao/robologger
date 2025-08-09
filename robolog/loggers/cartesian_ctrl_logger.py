from robolog.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt

class CartesianCtrlLogger(BaseLogger):
    def __init__(self, name: str, root_dir: str, project_name: str, task_name: str, run_name: str, attr: dict):
        super().__init__(name, root_dir, project_name, task_name, run_name, attr)

    def _init_storage(self):
        # TODO (jinyun): initialize zarr store
        ...

    def _close_storage(self):
        pass

    def log_state(
        self,
        *,
        state_timestamp: float,
        state_pos_xyz: npt.NDArray[np.float32],
        state_quat_wxyz: npt.NDArray[np.float32],
    ):
        
        # TODO (jinyun): log cartesian data
        ...

    def log_target(
        self,
        *,
        target_timestamp: float,
        target_pos_xyz: npt.NDArray[np.float32],
        target_quat_wxyz: npt.NDArray[np.float32],
    ):
        
        # TODO (jinyun): log cartesian data
        ...
