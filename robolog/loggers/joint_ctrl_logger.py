from typing import Optional
from robolog.loggers.base_logger import BaseLogger
import numpy as np
import numpy.typing as npt

class JointCtrlLogger(BaseLogger):
    def __init__(self, root_dir: str, project_name: str, task_name: str, run_name: str):
        super().__init__(root_dir, project_name, task_name, run_name)

    def _init_storage(self):
        # TODO (jinyun): initialize zarr store
        ...

    def _close_storage(self):
        pass

    def log(
        self,
        *,
        timestamp: float,
        state_joint_pos: npt.NDArray[np.float32],
        target_joint_pos: npt.NDArray[np.float32],
        state_joint_vel: Optional[npt.NDArray[np.float32]],
        state_joint_torque: Optional[npt.NDArray[np.float32]],
    ):
        super().log(timestamp=timestamp)
        # TODO (jinyun): log cartesian data
        ...

