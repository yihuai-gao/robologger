# Multiton pattern for loggers
from typing import Optional
from robologger.loggers.base_logger import LoggerType
from robologger.loggers.joint_ctrl_logger import JointCtrlLogger
from robologger.loggers.video_logger import VideoLogger
from robologger.loggers.cartesian_ctrl_logger import CartesianCtrlLogger
from robologger.loggers.sensor_logger import SensorLogger

def get_main_logger(
    name: str,
    root_dir: Optional[str],
    project_name: Optional[str],
    task_name: Optional[str],
    run_name: Optional[str],
    attr: Optional[dict],
):
    pass

def get_video_logger(
    name: str,
    attr: Optional[dict],
) -> VideoLogger:
    ...

def get_joint_ctrl_logger(
    name: str,
    attr: Optional[dict],
) -> JointCtrlLogger:
    ...

def get_cartesian_ctrl_logger(
    name: str,
    attr: Optional[dict],
) -> CartesianCtrlLogger:
    ...

def get_sensor_logger(
    name: str,
    attr: Optional[dict],
) -> SensorLogger:
    ...