from robologger.loggers.main_logger import MainLogger
from robologger.classes import Morphology
import time

logger = MainLogger(
    name="main",
    root_dir="data",
    project_name="test_mobile_base_all_modes",
    task_name="test_multiprocess",
    run_name="test_run",
    logger_endpoints={
        "mobile_base_pose_only": "tcp://localhost:55557",
        "mobile_base_with_velocity": "tcp://localhost:55558",
        "mobile_base_state_vel_only": "tcp://localhost:55559",
        "mobile_base_velocity_only": "tcp://localhost:55560",
        "mobile_base_mixed": "tcp://localhost:55561",
    },
    morphology=Morphology.WHEEL_BASED_SINGLE_ARM,
)

print("[MainLogger] Starting comprehensive multiprocess test for MobileBaseLogger")
print("[MainLogger] Testing all logging modes:")
print("  - pose_only: logs pose only")
print("  - with_velocity: logs pose + velocity for both state and target")
print("  - state_vel_only: logs state velocity, no target velocity")
print("  - velocity_only: logs velocity only (NO pose)")
print("  - mixed: state has pose+vel, target has vel only")
print("[MainLogger] Waiting for child loggers to connect...")
time.sleep(2)

try:
    for i in range(3):
        print(f"\n[MainLogger] === Recording session {i+1}/3 ===")
        logger.start_recording()

        print(f"[MainLogger] Recording for 5 seconds...")
        time.sleep(5)

        print(f"[MainLogger] Stopping recording...")
        logger.stop_recording()

        print(f"[MainLogger] Waiting before next session...")
        time.sleep(2)

    print("\n[MainLogger] All test sessions completed successfully!")

except KeyboardInterrupt:
    print("\n[MainLogger] Test interrupted by user")
    if logger._is_recording:
        logger.stop_recording()
except Exception as e:
    print(f"\n[MainLogger] Error during test: {e}")
    if logger._is_recording:
        logger.stop_recording()
    raise
