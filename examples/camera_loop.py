"""
Camera loop with robologger

Demonstrates how to integrate VideoLogger into a camera capture loop:
1. Define camera_configs dict with camera names, types (rgb/depth), and resolutions (for metadata)
2. Initialize VideoLogger with a valid CameraName (e.g., right_wrist_camera_0)
3. Run capture loop at desired frequency
4. Check recording state with update_recording_state()
5. Log frames using log_frame() with camera_name, timestamp, and frame data

The logger handles video encoding (MP4) and timestamp storage automatically.
"""
import time
import colorsys
import numpy as np
from robologger.loggers.video_logger import VideoLogger

# Camera configuration: Define all cameras this logger will handle
# Each camera needs: type (rgb/depth), width, height, fps
CAMERA_CONFIG = {
    "main": {
        "type": "rgb",         # RGB image (uint8, H×W×3)
        "width": 960,
        "height": 720,
        "fps": 30,
    },
    "ultrawide": {
        "type": "rgb",
        "width": 960,
        "height": 720,
        "fps": 30,
    },
    "depth": {
        "type": "depth",       # Depth map (float32, H×W, values in meters)
        "width": 320,
        "height": 240,
        "fps": 30,
    },
}

def main():
    # Initialize VideoLogger
    logger = VideoLogger(
        name="right_wrist_camera_0",           # Must match CameraName enum (see utils/classes.py)
                                                # Zero-indexed if multiple cameras at same position
                                                # e.g., right_wrist_camera_0, right_wrist_camera_1, ...
        endpoint="tcp://localhost:55555",      # RMQ endpoint for main process to connect
        attr={"camera_configs": CAMERA_CONFIG} # Camera configurations defined above
    )

    fps = CAMERA_CONFIG["main"]["fps"]
    dt = 1.0 / fps
    frame_count = 0  # Track frames for rainbow color cycling

    print("Camera loop running. Press Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.monotonic()

            # Capture timestamp for this frame
            timestamp = time.monotonic()

            # Check if main process has requested recording (with the main logger initialized in main_process.py)
            # This updates internal state and returns True if recording is active
            if logger.update_recording_state():
                # Log frames for each camera
                for cam_name, config in CAMERA_CONFIG.items():
                    h, w = config["height"], config["width"]

                    # Simulate camera capture (replace with actual camera API)
                    # Use rainbow colors for continuous frames to verify video compression and logging
                    if config["type"] == "rgb":
                        # Cycle through rainbow colors (hue 0-1)
                        hue = (frame_count % 180) / 180.0
                        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                        # Create solid color frame (demonstrates compression advantage)
                        image = np.full((h, w, 3), [int(r*255), int(g*255), int(b*255)], dtype=np.uint8)
                    elif config["type"] == "depth":
                        # Cycle depth from 0.1m to 3.5m
                        depth_value = 0.1 + (frame_count % 180) / 180.0 * (3.5 - 0.1)
                        image = np.full((h, w), depth_value, dtype=np.float32)

                    # Log the frame (encodes to MP4 and stores timestamp)
                    logger.log_frame(
                        camera_name=cam_name,  # Must match key in CAMERA_CONFIG
                        timestamp=timestamp,    # Monotonic time in seconds
                        frame=image            # numpy array (rgb: uint8, depth: float32)
                    )

                frame_count += 1

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()