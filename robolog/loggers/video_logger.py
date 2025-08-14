import os
import subprocess
from typing import Dict, Any

import cv2
import numpy as np
import numpy.typing as npt
import zarr
from loguru import logger

from robolog.utils.huecodec import depth2rgb, EncoderOpts
from robolog.loggers.base_logger import BaseLogger

class VideoLogger(BaseLogger):
    def __init__(
        self,
        name: str,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        attr: dict,
        depth_enc_mode: str = "hue_codec",
        depth_range: tuple[float, float] = (0.02, 4.0),
    ):
        super().__init__(name, root_dir, project_name, task_name, run_name, attr)
        self.ffmpeg_processes: Dict[str, subprocess.Popen] = {}

        self._validate_camera_config(attr)

        assert depth_enc_mode in ["gray_scale", "hue_codec", "hue_codec_inv"]
        # TODO: double check mode
        self.depth_enc_mode = depth_enc_mode
        self.depth_range = depth_range
        self.hue_opts = EncoderOpts(use_lut=True)

    def _validate_camera_config(self, attr: dict) -> None:
        """Validate camera config"""
        if "camera_config" not in attr:
            raise ValueError("Missing 'camera_config' in attr")
        if not isinstance(attr["camera_config"], dict):
            raise ValueError("'camera_config' must be a dictionary")
        if not attr["camera_config"]:
            raise ValueError("'camera_config' cannot be empty")

        required_keys = ["width", "height", "fps", "type"]
        for cam_name, config in attr["camera_config"].items():
            if not isinstance(config, dict):
                raise ValueError(f"Camera config for '{cam_name}' must be a dictionary")
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in camera config for '{cam_name}'")
            if config["type"] not in ["rgb", "depth"]:
                raise ValueError(f"Camera type for '{cam_name}' must be 'rgb' or 'depth', got '{config['type']}'")

    def _init_storage(self):
        """Initialize storage"""
        if not hasattr(self, 'run_dir') or not self.run_dir:
            raise RuntimeError("BaseLogger not properly initialized: missing run_dir")
        if not hasattr(self, 'episode_idx') or self.episode_idx < 0:
            raise RuntimeError("BaseLogger not properly initialized: invalid episode_idx")

        episode_dir = os.path.join(self.run_dir, f"episode_{self.episode_idx:06d}") # pad with zeros to 6 digits
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
            logger.info(f"[{self.name}] Created episode directory: {episode_dir}")

        zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")
        self.zarr_group = zarr.open_group(zarr_path, mode="w")
        logger.info(f"[{self.name}] Initialized zarr group: {zarr_path}")

        # timestamp arrays for each camera
        # NOTE: camera_config is stored in attr as a (nested) dict
        for cam_name in self.attr["camera_config"].keys():
            self.zarr_group.create_dataset(
                f"{cam_name}_timestamps",
                shape=(0,),
                chunks=(1000,),
                dtype=np.float64,
            )
            logger.info(f"[{self.name}] Created timestamp array for camera: {cam_name}")
        
        # initialize ffmpeg processes for each camera
        for cam_name, config in self.attr["camera_config"].items():
            mp4_file_path = os.path.join(episode_dir, f"{cam_name}.mp4")
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
                  "h264_nvenc",
                  "-pix_fmt",
                  "yuv420p",
                  "-preset",
                  "fast",
                  "-b:v",
                  "5M",
                  mp4_file_path,
              ]
            self.ffmpeg_processes[cam_name] = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE
            )
            logger.info(f"[{self.name}] Initialized ffmpeg process for camera: {cam_name}")


    def _close_storage(self):
        """Close storage"""
        for cam_name, process in self.ffmpeg_processes.items():
              logger.info(f"Stopping FFmpeg process for '{cam_name}'...")
              if process.stdin:
                  process.stdin.close()
              try:
                  process.wait(timeout=10)
              except subprocess.TimeoutExpired:
                  logger.warning(f"FFmpeg process for '{cam_name}' did not terminate gracefully, forcing kill")
                  process.kill()
                  process.wait()
              logger.info(f"FFmpeg process for '{cam_name}' stopped.")

        self.ffmpeg_processes.clear()
        self.zarr_group = None

    def _encode_depth_to_rgb(self, depth_frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Encode depth frame to RGB using the same encoding as iPhone implementation"""
        if self.depth_enc_mode == "hue_codec":
            depth_clipped = np.clip(depth_frame, self.depth_range[0], self.depth_range[1])
            depth_rgb_float = depth2rgb(
                depth_clipped, self.depth_range, inv_depth=False, opts=self.hue_opts
            )
            depth_rgb_uint8 = (depth_rgb_float * 255).astype(np.uint8)
            return depth_rgb_uint8  # RGB

        elif self.depth_enc_mode == "hue_codec_inv":
            depth_clipped = np.clip(depth_frame, self.depth_range[0], self.depth_range[1])
            depth_rgb_float = depth2rgb(
                depth_clipped, self.depth_range, inv_depth=True, opts=self.hue_opts
            )
            depth_rgb_uint8 = (depth_rgb_float * 255).astype(np.uint8)
            return depth_rgb_uint8  # RGB

        elif self.depth_enc_mode == "gray_scale":
            depth_normalized = np.clip(depth_frame / self.depth_range[1], 0, 1)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            return depth_uint8[..., np.newaxis].repeat(3, axis=2)  # grayscale is same in RGB/BGR

        else:
            raise ValueError(f"Invalid depth encoding mode: {self.depth_enc_mode}")

    def log_frame(
        self,
        *,
        camera_name: str,
        timestamp: float,
        frame: npt.NDArray[np.uint8],  # RGB 
    ):
        if camera_name not in self.attr["camera_config"]:
            raise ValueError(f"Camera '{camera_name}' not found in camera config")
        
        if self.zarr_group is None:
            raise RuntimeError("Storage not initialized. Call start_episode() first.")
        
        config: dict = self.attr["camera_config"][camera_name]

        if config["type"] == "rgb":
            expected_shape = (config["height"], config["width"], 3) # HWC

            assert frame.dtype == np.uint8, f"RGB frame must be uint8, got {frame.dtype}"
            assert len(frame.shape) == 3, f"RGB frame must be HWC (3D), got shape {frame.shape}"
            assert frame.shape[2] == 3, f"RGB frame must have 3 channels, got {frame.shape[2]}"

            if frame.shape != expected_shape:
                raise ValueError(f"RGB frame shape mismatch for camera '{camera_name}'. "
                                f"Expected {expected_shape}, got {frame.shape}")
            frame_rgb = frame  # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR

        elif config["type"] == "depth":
            expected_shape = (config["height"], config["width"])
            assert len(frame.shape) == 2, f"Depth frame must be 2D (HW), got shape {frame.shape}"
            if frame.shape != expected_shape:
                raise ValueError(f"Depth frame shape mismatch for camera '{camera_name}'. "
                                f"Expected {expected_shape}, got {frame.shape}")
            frame_rgb = self._encode_depth_to_rgb(frame.astype(np.float32))  # RGB

            assert len(frame_rgb.shape) == 3, f"Encoded depth RGB must be HWC (3D), got shape {frame_rgb.shape}"
            assert frame_rgb.shape[2] == 3, f"Encoded depth RGB must have 3 channels, got {frame_rgb.shape[2]}"

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR

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
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.write(frame_bgr.tobytes())
                    process.stdin.flush()
                except BrokenPipeError:
                    logger.error(f"[{self.name}] FFmpeg process for '{camera_name}' closed unexpectedly")
                    # gracefully close the process
                    if process.stdin:
                        process.stdin.close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    # remove from active processes
                    del self.ffmpeg_processes[camera_name]
                    raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' closed unexpectedly")
        else:
            logger.warning(f"[{self.name}] FFmpeg process for '{camera_name}' not available")
            raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' not available")
        
    def log_frames(self, frame_dict: Dict[str, Dict[str, Any]]):
        """Log frames for all cameras with individual timestamps
        
        Expected frame_dict format:
        {
            "camera_name": {
                "frame": npt.NDArray[np.uint8],  # RGB format, HWC shape expected
                "timestamp": float
            }
        }
        """
        # Validate that all frame_dict keys exist in camera_config
        for cam_name in frame_dict.keys():
            if cam_name not in self.attr["camera_config"]:
                raise ValueError(f"Camera '{cam_name}' not found in camera config")
        
        # log_frame will handle all validation, so just call it
        for cam_name, frame_data in frame_dict.items():
            self.log_frame(
                  camera_name=cam_name,
                  timestamp=frame_data['timestamp'],
                  frame=frame_data['frame']
            )



        
