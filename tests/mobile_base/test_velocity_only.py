import time
from robologger.loggers.mobile_base_logger import MobileBaseLogger
import numpy as np

logger = MobileBaseLogger(
    name="mobile_base_velocity_only",
    endpoint="tcp://localhost:55560",
    attr={
        "robot_type": "mobile_base",
        "test_mode": "velocity_only"
    },
    log_state_pose=False,
    log_state_velocity=True,
    log_target_pose=False,
    log_target_velocity=True
)

print(f"[{logger.name}] Starting test - velocity only mode (no pose)")

sim_time = 0.0
while True:
    is_recording = logger.update_recording_state()
    if is_recording:
        sim_time += 0.1

        state_vx = 0.5 * np.cos(sim_time * 0.5)
        state_vy = -0.5 * np.sin(sim_time * 0.5)
        state_vyaw = 0.2

        target_vx = state_vx + 0.05
        target_vy = state_vy + 0.05
        target_vyaw = state_vyaw + 0.05

        logger.log_state(
            state_timestamp=time.time(),
            state_velocity=np.array([state_vx, state_vy, state_vyaw])
        )
        logger.log_target(
            target_timestamp=time.time(),
            target_velocity=np.array([target_vx, target_vy, target_vyaw])
        )

        if logger.state_count % 100 == 0:
            print(f"[{logger.name}] Logged {logger.state_count} states, {logger.target_count} targets")

    time.sleep(0.01)
