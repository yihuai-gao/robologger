import time
from robologger.loggers.ctrl_logger import RobotCtrlLogger
import numpy as np

logger = RobotCtrlLogger(
    name="right_arm",  # Must use valid RobotName enum
    endpoint="tcp://localhost:55556",
    attr={
        "test_attr": "test_value",
    },
    log_eef_pose=True,
    log_joint_pos=False,
    target_type="eef_pose",
    joint_units=None
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