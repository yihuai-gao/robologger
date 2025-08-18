import time
from robologger.loggers.video_logger import VideoLogger
import numpy as np

logger = VideoLogger(
    name="test_video_logger",
    endpoint="tcp://localhost:55555",
    attr={
        "camera_configs": {
            "test_camera": {
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "type": "rgb",
            }
        }
    },
)

while True:
    print(logger.is_recording)
    if logger.is_recording:
        logger.log_frame(
            camera_name="test_camera",
            timestamp=time.time(),
            frame=np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
        )
    time.sleep(0.1)