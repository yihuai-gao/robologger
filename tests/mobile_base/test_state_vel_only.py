import time
from robologger.loggers.mobile_base_logger import MobileBaseLogger
import numpy as np

logger = MobileBaseLogger(
    name="mobile_base_state_vel_only",
    endpoint="tcp://localhost:55559",
    attr={
        "robot_type": "mobile_base",
        "test_mode": "state_velocity_only"
    },
    log_state_pose=True,
    log_state_velocity=True,
    log_target_pose=True,
    log_target_velocity=False
)

print(f"[{logger.name}] Starting test - state velocity only mode")

sim_time = 0.0
while True:
    is_recording = logger.update_recording_state()
    if is_recording:
        sim_time += 0.1

        state_x = np.sin(sim_time * 0.5)
        state_y = np.cos(sim_time * 0.5)
        state_yaw = sim_time * 0.2

        state_vx = 0.5 * np.cos(sim_time * 0.5)
        state_vy = -0.5 * np.sin(sim_time * 0.5)
        state_vyaw = 0.2

        target_x = state_x + 0.1
        target_y = state_y + 0.1
        target_yaw = state_yaw + 0.1

        logger.log_state(
            state_timestamp=time.time(),
            state_pose=np.array([state_x, state_y, state_yaw]),
            state_velocity=np.array([state_vx, state_vy, state_vyaw])
        )
        logger.log_target(
            target_timestamp=time.time(),
            target_pose=np.array([target_x, target_y, target_yaw])
        )

        if logger.state_count % 100 == 0:
            print(f"[{logger.name}] Logged {logger.state_count} states, {logger.target_count} targets")

    time.sleep(0.01)
