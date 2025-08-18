from robologger.loggers.main_logger import MainLogger
import time

logger = MainLogger(
    name="main",
    root_dir="data",
    project_name="test_project",
    task_name="test_task",
    run_name="test_run",
    logger_endpoints={
        "test_video_logger": "tcp://localhost:55555",
        "test_cartesian_ctrl_logger": "tcp://localhost:55556",
    },
)

logger.start_recording()

time.sleep(10)

logger.stop_recording()