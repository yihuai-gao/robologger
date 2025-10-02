from robologger.loggers.main_logger import MainLogger
import time

logger = MainLogger(
    name="main",
    root_dir="data",
    project_name="test_project",
    task_name="test_task",
    run_name="test_run",
    logger_endpoints={
        "right_wrist_camera_0": "tcp://localhost:55555",
        "right_arm": "tcp://localhost:55556",
    },
)

while True:
    logger.start_recording()

    time.sleep(10)

    logger.stop_recording()
    
    time.sleep(0.1)