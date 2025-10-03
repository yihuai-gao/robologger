import os
import subprocess
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import zarr
from loguru import logger

from robologger.utils.huecodec import depth2logrgb, EncoderOpts
from robologger.loggers.base_logger import BaseLogger

class VideoLogger(BaseLogger):
    """Logger for video data from multi-camera subsystems.
    
    Each VideoLogger handles multiple camera streams with custom names stored in zarr file attributes.
    Multiple VideoLoggers can be instantiated for each camera subsystem. For example, if you have 
    3 iPhones mounted on the right wrist, each with 3 camera types (main_rgb, ultrawide_rgb, depth, etc.),
    you would create 3 VideoLoggers with names like: right_wrist_camera_0, right_wrist_camera_1, 
    right_wrist_camera_2 - one VideoLogger for each iPhone on the same wrist.
    
    Naming Convention:
    - Must use CameraName enum values with zero-indexed suffixes
    - Multiple loggers of same enum type must use continuous zero-indexed naming
    """
    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
        codec: str = "h264_nvenc", # XXX: can use av1_nvenc if supported
        depth_range: Tuple[float, float] = (0.0, 4.0), # TODO: choose a proper range
    ):
        """Initialize video logger with camera configurations."""
        super().__init__(name, endpoint, attr)
        self.ffmpeg_processes: Dict[str, subprocess.Popen[bytes]] = {}

        self._validate_camera_config(attr)
        self.depth_range = depth_range
        self.hue_opts = EncoderOpts(use_lut=True)
        self.codec = codec


    def _validate_camera_config(self, attr: Dict[str, Any]) -> None:
        """Validate camera configuration dictionary."""
        if "camera_configs" not in attr:
            raise ValueError("Missing 'camera_configs' in attr")
        if not isinstance(attr["camera_configs"], dict):
            raise ValueError("'camera_configs' must be a dictionary")
        if not attr["camera_configs"]:
            raise ValueError("'camera_configs' cannot be empty")

        required_keys = ["width", "height", "fps", "type"]
        for cam_name, config in attr["camera_configs"].items():
            if not isinstance(config, dict):
                raise ValueError(f"Camera config for '{cam_name}' must be a dictionary")
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in camera config for '{cam_name}'")
            if config["type"] not in ["rgb", "depth"]:
                raise ValueError(f"Camera type for '{cam_name}' must be 'rgb' or 'depth', got '{config['type']}'")

    def _init_storage(self):
        """Initialize zarr storage and FFmpeg processes for video recording."""
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")

        
        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")
        self.zarr_group = zarr.open_group(zarr_path, mode="w")
        logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")

        # timestamp arrays for each camera
        # NOTE: camera_config is stored in attr as a (nested) dict
        for cam_name in self.attr["camera_configs"].keys():
            self.zarr_group.create_dataset(
                f"{cam_name}_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float32,
            )
            logger.info(f"[{self.name}] Created timestamp array for camera: {cam_name}")
        
        try:
            for cam_name, config in self.attr["camera_configs"].items():
                mp4_file_path = os.path.join(zarr_path, f"{cam_name}.mp4")
                ffmpeg_cmd = [
                      "ffmpeg",
                      "-y",
                      "-f",
                      "rawvideo",
                      "-vcodec",
                      "rawvideo",
                      "-pix_fmt",
                      "bgr24",
                      "-s",
                      f"{config['width']}x{config['height']}",
                      "-r",
                      str(config["fps"]),
                      "-i",
                      "-",
                      "-c:v",
                      self.codec,
                      "-pix_fmt",
                      "yuv420p",
                      "-preset",
                      "fast",
                      "-b:v",
                      "5M",
                      mp4_file_path,
                  ]
                self.ffmpeg_processes[cam_name] = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # XXX: Remove to see ffmpeg encoding output
                    stdout=subprocess.DEVNULL   # XXX: Remove to see ffmpeg encoding output
                )
                logger.info(f"[{self.name}] Initialized ffmpeg process for camera: {cam_name}")
        except Exception as e:
            for cam_name, process in self.ffmpeg_processes.items():
                self._close_ffmpeg_process(cam_name, process)
            self.ffmpeg_processes.clear()
            raise RuntimeError(f"Failed to initialize FFmpeg processes: {e}") from e


    def _close_ffmpeg_process(self, cam_name: str, process: subprocess.Popen, timeout: int = 10) -> None:
        """Gracefully close an FFmpeg process."""
        logger.info(f"Stopping FFmpeg process for '{cam_name}'...")
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg process for '{cam_name}' did not terminate gracefully, forcing kill")
            process.kill()
            process.wait()
        logger.info(f"FFmpeg process for '{cam_name}' stopped.")

    def _close_storage(self):
        """Close FFmpeg processes and zarr storage."""
        if self.zarr_group is None:
            logger.error(f"[{self.name}] Zarr group is not initialized but _close_storage() is called in video logger")
            return
        
        for cam_name, process in self.ffmpeg_processes.items():
            self._close_ffmpeg_process(cam_name, process)

        self.ffmpeg_processes.clear()
        self.zarr_group = None

    def log_frame(
        self,
        *,
        camera_name: str,
        timestamp: float,
        frame: npt.NDArray[Any],  # RGB uint8 or depth float32
    ):
        """Log single video frame with timestamp."""
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received frame command")
            return
        
        if self.zarr_group is None:
            raise ValueError("Storage not initialized. Please call start_episode() before logging frames to make sure the zarr group is initialized.")
        
        if camera_name not in self.attr["camera_configs"]:
            raise ValueError(f"Camera '{camera_name}' not found in camera config")
        
        config: dict = self.attr["camera_configs"][camera_name]

        if config["type"] == "rgb":
            expected_shape = (config["height"], config["width"], 3)
            
            assert frame.dtype == np.uint8, f"RGB frame must be uint8, got {frame.dtype}"
            assert len(frame.shape) == 3, f"RGB frame must be HWC (3D), got shape {frame.shape}"
            assert frame.shape[2] == 3, f"RGB frame must have 3 channels, got {frame.shape[2]}"

            if frame.shape != expected_shape:
                raise ValueError(f"RGB frame shape mismatch for camera '{camera_name}'. "
                                f"Expected {expected_shape}, got {frame.shape}")
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif config["type"] == "depth":
            expected_shape = (config["height"], config["width"])
            
            assert frame.dtype in [np.float16, np.float32, np.float64], f"Depth frame must be float16 or float32 or float64, got {frame.dtype}"
            assert len(frame.shape) == 2, f"Depth frame must be HW (2D), got shape {frame.shape}"
            
            if frame.shape != expected_shape:
                raise ValueError(f"Depth frame shape mismatch for camera '{camera_name}'. "
                                f"Expected {expected_shape}, got {frame.shape}")
            
            frame_rgb = depth2logrgb(frame, self.depth_range, opts=self.hue_opts)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow("frame_bgr", frame_bgr)
            cv2.waitKey(1)

        else:
            raise ValueError(f"Unknown camera type: {config['type']}")

        timestamp_dataset = self.zarr_group[f"{camera_name}_timestamps"]
        assert isinstance(timestamp_dataset, zarr.Array), "Timestamp dataset must be a zarr.Array"
        original_shape = timestamp_dataset.shape
        new_shape = (original_shape[0] + 1, *original_shape[1:])
        timestamp_dataset.resize(new_shape)
        timestamp_dataset[-1] = timestamp

        if camera_name in self.ffmpeg_processes:
            process = self.ffmpeg_processes[camera_name]
            if process.stdin:
                try:
                    process.stdin.write(frame_bgr.tobytes())
                    process.stdin.flush()
                except BrokenPipeError:
                    self._close_ffmpeg_process(camera_name, process, timeout=5)
                    del self.ffmpeg_processes[camera_name] # remove from active processes dict
                    raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' closed unexpectedly")
        else:
            logger.warning(f"[{self.name}] FFmpeg process for '{camera_name}' not available")
            raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' not available")
        
    def log_frames(self, frame_dict: Dict[str, Dict[str, Any]]):
        """Log frames for multiple cameras with individual timestamps.
        
        Expected frame_dict format:
        {
            "camera_name": {
                "frame": Union[npt.NDArray[np.uint8], npt.NDArray[np.float32]],  # RGB uint8 (HWC) or depth float32 (HW)
                "timestamp": float
            }
        }
        """
        # Validate that all frame_dict keys exist in camera_config
        for cam_name in frame_dict.keys():
            if cam_name not in self.attr["camera_configs"]:
                raise ValueError(f"Camera '{cam_name}' not found in camera config")
        
        # log_frame will handle all validation, so just call it
        for cam_name, frame_data in frame_dict.items():
            self.log_frame(
                  camera_name=cam_name,
                  timestamp=frame_data['timestamp'],
                  frame=frame_data['frame']
            )



        
