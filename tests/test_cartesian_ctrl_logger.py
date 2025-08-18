import time
from robologger.loggers.cartesian_ctrl_logger import CartesianCtrlLogger
import numpy as np

logger = CartesianCtrlLogger(
    name="test_cartesian_ctrl_logger",
    endpoint="tcp://localhost:55556",
    attr={
        "test_attr": "test_value",
    },
)

while True:
    print(logger.is_recording)
    if logger.is_recording:
        logger.log_state(
            state_timestamp=time.time(),
            state_pos_xyz=np.array([0.0, 0.0, 0.0]),
            state_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        logger.log_target(
            target_timestamp=time.time(),
            target_pos_xyz=np.array([0.0, 0.0, 0.0]),
            target_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    time.sleep(0.1)