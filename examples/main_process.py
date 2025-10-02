"""
Main process that manages recording with robologger

The main process orchestrates data collection by:
1. Running separately from control loops (camera, robot, gripper)
2. Communicating with control loops via RobotMQ to start/stop recording
3. Managing episode metadata and coordinating data storage
4. Allowing operator to control when to record data

Control loops run continuously generating data, but only log when recording is active.

Keyboard controls:
- Terminal input (stdin):
  - 's': Start recording
  - 'e': End/stop recording
  - 'd': Delete last episode
  - 'q' or Ctrl+C or Ctrl+\\: Quit
"""

import sys
import select
import termios
import tty
import threading
import time

from robologger.loggers.main_logger import MainLogger
from robologger.utils.classes import Morphology

# Global state for keyboard input
terminal_key_name: str | None = None
continue_running: bool = True
pause_keyboard: bool = False


def read_char_with_timeout(timeout=0.1):
    """Read a single character from stdin with timeout."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def keyboard_thread():
    """Thread to continuously read terminal input."""
    global continue_running, terminal_key_name, pause_keyboard
    while continue_running:
        if pause_keyboard:
            time.sleep(0.01)
            continue
        terminal_key_name = read_char_with_timeout()
        if terminal_key_name in ("q", "\x03", "\x1c"):  # q, Ctrl+C, Ctrl+\
            print("\nQuitting...")
            continue_running = False


def main():
    global terminal_key_name, continue_running, pause_keyboard

    # Logger endpoints: Maps logger names to their RobotMQ addresses
    # Must match the endpoints used by individual control loops (camera, robot, gripper)
    logger_endpoints = {
        "right_wrist_camera_0": "tcp://localhost:55555",  # Camera loop endpoint
        "right_arm": "tcp://localhost:55556",              # Robot control loop endpoint
        "right_end_effector": "tcp://localhost:55557",     # Gripper control loop endpoint
    }

    # Initialize main logger
    # success_config options:
    #   - "none": Does not set is_successful (no prompt, no value assigned)
    #   - "input_true": Prompt user with [Y/n], defaults to successful
    #   - "input_false": Prompt user with [y/N], defaults to failed
    #   - "hardcode_true": Always mark episodes as successful (no prompt)
    #   - "hardcode_false": Always mark episodes as failed (no prompt)
    main_logger = MainLogger(
        name="main_logger",
        root_dir="data",
        project_name="demo_project",
        task_name="demo_task",
        run_name="run_001",
        logger_endpoints=logger_endpoints,
        morphology=Morphology.SINGLE_ARM,
        success_config="input_true",
    )

    print("\n" + "="*60)
    print("  ROBOLOGGER - Main Process")
    print("="*60)
    print("\nCommands:")
    print("  's' - Start recording")
    print("  'e' - End recording")
    print("  'd' - Delete last episode")
    print("  'q' - Quit (or Ctrl+C/Ctrl+\\)")
    print("="*60 + "\n")

    # Start keyboard input thread (daemon=True makes it exit when main thread exits)
    threading.Thread(target=keyboard_thread, daemon=True).start()

    try:
        while continue_running:
            if terminal_key_name is not None:
                key = terminal_key_name
                terminal_key_name = None

                if key in ("\x03", "\x1c", "q"):  # Ctrl+C, Ctrl+\, q
                    break
                if key in ("\r", "\n"):  # Ignore Enter key
                    continue

                # Print separator and handle command
                print(f"\n{'─'*60}")
                if key == "s":
                    print("Starting recording...")
                    try:
                        episode_idx = main_logger.start_recording()
                        print(f"Recording episode {episode_idx}")
                    except Exception as e:
                        print(f"Failed to start recording: {e}")
                elif key == "e":
                    print("Stopping recording...")
                    pause_keyboard = True  # Pause for user input prompt
                    time.sleep(0.02)  # Let keyboard thread pause
                    episode_idx = main_logger.stop_recording()
                    pause_keyboard = False  # Resume keyboard thread
                    if episode_idx is not None:
                        print(f"Stopped recording episode {episode_idx}")
                elif key == "d":
                    print("Deleting last episode...")
                    episode_idx = main_logger.delete_last_episode()
                    if episode_idx is not None:
                        print(f"Deleted episode {episode_idx}")
                print(f"{'─'*60}\n")
            else:
                time.sleep(0.01)  # Avoid busy-wait

    except KeyboardInterrupt:
        pass
    finally:
        continue_running = False
        print("\nStopped")


if __name__ == "__main__":
    main()