#!/usr/bin/env python3
"""
Robot episode visualizer for UMI and RoboLogger data formats.

Provides visualization of robot trajectories, gripper states, and camera feeds
using Rerun. Supports both UMI format (UR5 + WSG50 + GoPro) and RoboLogger format.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

import zarr
import numpy as np
import cv2
import rerun as rr

from robologger.scripts.rerun_visualization.base_visualizer import BaseVisualizer


class UMIVisualizer(BaseVisualizer):
    """
    Robot episode visualizer for manipulation data.

    Visualizes robot poses, gripper states, trajectories, and synchronized
    video feeds using Rerun. Supports UMI and RoboLogger data formats.
    """
    
    # Camera intrinsic parameters for GoPro
    DEFAULT_CAMERA_FX = 280.0
    DEFAULT_CAMERA_FY = 280.0
    DEFAULT_CAMERA_CX = 640.0
    DEFAULT_CAMERA_CY = 360.0
    DEFAULT_CAMERA_WIDTH = 1280
    DEFAULT_CAMERA_HEIGHT = 720
    DEFAULT_CAMERA_PLANE_WIDTH_M = 0.1  # physical width to render the pinhole plane in 3D
    
    # UMI camera mounting configuration (meters)
    DEFAULT_TCP_OFFSET = 0.13
    
    def __init__(self,
                 episode_data: Union[str, Path, Dict[str, Any]],
                 max_frames: int = 500,
                 video_rate: int = 3,
                 dataset_type: str = "auto"):
        """
        Initialize robot episode visualizer.

        Args:
            episode_data: Path to episode directory or pre-loaded data dict
            max_frames: Maximum frames to process for performance optimization
            video_rate: Video frame sampling rate (process every Nth frame)
            dataset_type: Dataset format - "auto", "umi", or "robologger"

        Raises:
            ValueError: If episode_data format is invalid
        """
        super().__init__()

        self.max_frames = max_frames
        self.video_rate = video_rate
        # Camera params will be adjusted based on the actual video resolution when available.
        self.camera_width = self.DEFAULT_CAMERA_WIDTH
        self.camera_height = self.DEFAULT_CAMERA_HEIGHT
        self.camera_plane_width_m = self.DEFAULT_CAMERA_PLANE_WIDTH_M
        self.camera_fx = self.DEFAULT_CAMERA_FX
        self.camera_fy = self.DEFAULT_CAMERA_FY
        self.camera_cx = self.DEFAULT_CAMERA_CX
        self.camera_cy = self.DEFAULT_CAMERA_CY

        if isinstance(episode_data, (str, Path)):
            episode_path = Path(episode_data)

            # Auto-detect or use specified format
            if dataset_type == "auto":
                is_robologger = self._is_robologger_format(episode_path)
            elif dataset_type == "robologger":
                is_robologger = True
            else:  # umi or other
                is_robologger = False

            if is_robologger:
                self.data = self._load_robologger_episode_data(episode_path)
            else:
                self.data = self._load_umi_episode_data(str(episode_data))
        elif isinstance(episode_data, dict):
            self.data = episode_data
        else:
            raise ValueError("episode_data must be a path string or pre-loaded data dict")

        self.trajectory_positions = []
        self.target_trajectory_positions = []
        self._update_camera_params_from_data()

    def _update_camera_params_from_data(self) -> None:
        """Update camera intrinsics/size to match the actual video resolution."""
        width = None
        height = None

        if self.data.get("is_zarr_video", False):
            video_array = self.data.get("video_array")
            if video_array is not None and video_array.shape and len(video_array.shape) >= 3:
                height = video_array.shape[1]
                width = video_array.shape[2]
        else:
            video_cap = self.data.get("video_cap")
            if video_cap is not None:
                cap_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if cap_w > 0 and cap_h > 0:
                    width, height = cap_w, cap_h

        if width is None or height is None or width == 0 or height == 0:
            # Fallback to defaults
            width = self.DEFAULT_CAMERA_WIDTH
            height = self.DEFAULT_CAMERA_HEIGHT

        self.camera_width = width
        self.camera_height = height

        # Scale intrinsics to the detected resolution while keeping reasonable defaults.
        scale_w = self.camera_width / self.DEFAULT_CAMERA_WIDTH
        scale_h = self.camera_height / self.DEFAULT_CAMERA_HEIGHT
        self.camera_fx = self.DEFAULT_CAMERA_FX * scale_w
        self.camera_fy = self.DEFAULT_CAMERA_FY * scale_h
        self.camera_cx = self.camera_width / 2.0
        self.camera_cy = self.camera_height / 2.0

    @staticmethod
    def _is_robologger_format(episode_path: Path) -> bool:
        """Check if path contains robologger format data."""
        # Robologger has episode_data.zarr with episode_N subdirectories
        zarr_path = episode_path / "episode_data.zarr"
        if zarr_path.exists():
            # Check for episode_0 subdirectory
            return (zarr_path / "episode_0").exists()
        return False
    
    def _load_umi_episode_data(self, episode_path: str) -> Dict[str, Any]:
        """
        Load UMI episode data from zarr files.
        
        Args:
            episode_path: Path to episode directory containing zarr files
            
        Returns:
            Dictionary containing all episode data
            
        Raises:
            ValueError: If video file cannot be opened or data is corrupted
        """
        # Load UR5 robot data
        ur5_data = zarr.open(f"{episode_path}/UR5.zarr", "r")
        ur5_poses = np.array(ur5_data["eef_poses"])
        ur5_timestamps = np.array(ur5_data["robot_timestamps"])
        
        # Load target poses if available
        ur5_targets = None
        ur5_target_timestamps = None
        if "eef_targets" in ur5_data.keys():
            ur5_targets = np.array(ur5_data["eef_targets"])
        if "target_timestamps" in ur5_data.keys():
            ur5_target_timestamps = np.array(ur5_data["target_timestamps"])
        elif "eef_target_timestamps" in ur5_data.keys():
            ur5_target_timestamps = np.array(ur5_data["eef_target_timestamps"])
        
        # Load WSG50 gripper data
        wsg50_data = zarr.open(f"{episode_path}/WSG50.zarr", "r")
        gripper_positions = np.array(wsg50_data["joint_positions"])
        gripper_timestamps = np.array(wsg50_data["robot_timestamps"])
        
        # Load gripper target commands if available
        gripper_targets = None
        gripper_target_timestamps = None
        if "joint_targets" in wsg50_data.keys():
            gripper_targets = np.array(wsg50_data["joint_targets"])
        if "target_timestamps" in wsg50_data.keys():
            gripper_target_timestamps = np.array(wsg50_data["target_timestamps"])

        # Load and validate video
        video_path = f"{episode_path}/GoPro.zarr/video.mp4"
        video_cap = cv2.VideoCapture(video_path)
        
        if not video_cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load camera timestamps with fallback
        video_timestamps = self._load_camera_timestamps(
            episode_path, ur5_timestamps, total_frames
        )
        
        return {
            'ur5_poses': ur5_poses,
            'ur5_timestamps': ur5_timestamps,
            'ur5_targets': ur5_targets,
            'ur5_target_timestamps': ur5_target_timestamps,
            'gripper_positions': gripper_positions,
            'gripper_timestamps': gripper_timestamps,
            'gripper_targets': gripper_targets,
            'gripper_target_timestamps': gripper_target_timestamps,
            'video_cap': video_cap,
            'video_array': None,  # Only for robologger format
            'video_timestamps': video_timestamps,
            'video_fps': fps,
            'total_frames': total_frames,
            'is_zarr_video': False  # UMI uses MP4
        }

    def _load_camera_timestamps(self, episode_path: str, ur5_timestamps: np.ndarray,
                               total_frames: int) -> np.ndarray:
        """
        Load camera timestamps with fallback to interpolated timestamps.

        Args:
            episode_path: Path to episode directory
            ur5_timestamps: Robot timestamps for synchronization
            total_frames: Total number of video frames

        Returns:
            Array of video timestamps synchronized to robot time
        """
        try:
            gopro_data = zarr.open(f"{episode_path}/GoPro.zarr", "r")
            camera_timestamps = np.array(gopro_data["timestamps"])

            if len(camera_timestamps) > 0:
                # Synchronize camera timestamps to robot time base
                return camera_timestamps - camera_timestamps[0] + ur5_timestamps[0]
            else:
                raise ValueError("Empty timestamps array")

        except Exception:
            # Fallback: linearly interpolate timestamps
            return np.linspace(ur5_timestamps[0], ur5_timestamps[-1], total_frames)

    def _load_robologger_episode_data(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Load robologger episode data from zarr archive.

        RoboLogger format:
        - episode_data.zarr/episode_N/robot0_tcp_xyz_wxyz: (N, 7) [x,y,z,w,x,y,z]
        - episode_data.zarr/episode_N/robot0_gripper_width: (N, 1)
        - episode_data.zarr/episode_N/action0_tcp_xyz_wxyz: (N, 7) commanded poses
        - episode_data.zarr/episode_N/action0_gripper_width: (N, 1) commanded gripper
        - episode_data.zarr/episode_N/third_person_camera/...: video data

        Args:
            dataset_path: Path to robologger dataset directory

        Returns:
            Dictionary containing all episode data in UMI format

        Raises:
            ValueError: If data cannot be loaded
        """
        zarr_path = dataset_path / "episode_data.zarr"

        # For now, load episode_0 (can be extended to load multiple episodes)
        episode_idx = 0
        episode_data = zarr.open(str(zarr_path / f"episode_{episode_idx}"), "r")

        # Load robot TCP poses (current state)
        robot_tcp = np.array(episode_data["robot0_tcp_xyz_wxyz"])  # (N, 7): [x,y,z,w,x,y,z]
        # Convert to UMI format [x,y,z,qw,qx,qy,qz] - robologger already uses this format
        ur5_poses = robot_tcp

        # Load action TCP poses (commanded targets)
        ur5_targets = None
        if "action0_tcp_xyz_wxyz" in episode_data.keys():
            ur5_targets = np.array(episode_data["action0_tcp_xyz_wxyz"])

        # Load gripper data
        gripper_positions = np.array(episode_data["robot0_gripper_width"])  # (N, 1)
        gripper_targets = None
        if "action0_gripper_width" in episode_data.keys():
            gripper_targets = np.array(episode_data["action0_gripper_width"])

        # Generate timestamps (robologger doesn't store timestamps explicitly)
        # Assume 10 Hz for robot data
        robot_freq = 10.0
        num_steps = len(ur5_poses)
        ur5_timestamps = np.arange(num_steps, dtype=np.float64) / robot_freq
        gripper_timestamps = ur5_timestamps.copy()
        ur5_target_timestamps = ur5_timestamps.copy() if ur5_targets is not None else None
        gripper_target_timestamps = ur5_timestamps.copy() if gripper_targets is not None else None

        # Load video data from third_person_camera
        # RoboLogger stores images as zarr arrays, not as MP4
        camera_data = episode_data["third_person_camera"]
        total_frames = camera_data.shape[0]

        # Assume 30 FPS for camera (typical for robot cameras)
        fps = 30.0

        # Generate video timestamps synchronized with robot data
        video_timestamps = np.linspace(ur5_timestamps[0], ur5_timestamps[-1], total_frames)

        # Create a dummy video capture object that we'll replace with zarr array access
        video_cap = None  # Will use camera_data array instead

        return {
            'ur5_poses': ur5_poses,
            'ur5_timestamps': ur5_timestamps,
            'ur5_targets': ur5_targets,
            'ur5_target_timestamps': ur5_target_timestamps,
            'gripper_positions': gripper_positions,
            'gripper_timestamps': gripper_timestamps,
            'gripper_targets': gripper_targets,
            'gripper_target_timestamps': gripper_target_timestamps,
            'video_cap': video_cap,
            'video_array': camera_data,  # Zarr array for robologger
            'video_timestamps': video_timestamps,
            'video_fps': fps,
            'total_frames': total_frames,
            'is_zarr_video': True  # Flag to indicate zarr-based video
        }
    
    def setup_static_components(self, **kwargs):
        """Setup static visualization components including coordinate systems and camera intrinsics."""
        # Setup coordinate systems
        rr.log("3D", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("2D", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Configure camera intrinsics
        self._setup_camera_intrinsics()
        
        # Configure layout blueprint
        blueprint = BaseVisualizer.create_robotics_blueprint(camera_names=["gopro"])
        rr.send_blueprint(blueprint)
        
        # Initialize plot series for gripper data
        self._initialize_plot_series()
    
    def _setup_camera_intrinsics(self):
        """Configure camera intrinsic parameters for both 2D and 3D views."""
        intrinsic_matrix = [
            [self.camera_fx, 0, self.camera_cx],
            [0, self.camera_fy, self.camera_cy],
            [0, 0, 1]
        ]

        # Set plane distance so the rendered pinhole plane matches the image aspect ratio
        # and stays at a sensible physical scale in the 3D view.
        plane_distance = (
            self.camera_plane_width_m
            * self.camera_fx
            / self.camera_width
        )

        pinhole_config = rr.Pinhole(
            image_from_camera=intrinsic_matrix,
            width=self.camera_width,
            height=self.camera_height,
            image_plane_distance=plane_distance
        )
        
        rr.log("2D/camera/gopro", pinhole_config, static=True)
        rr.log("3D/camera/gopro", pinhole_config, static=True)
    
    def _initialize_plot_series(self):
        """Initialize plot series for temporal data visualization."""
        rr.set_time("timestamp", timestamp=0.0)
        rr.log("2D/world/plot/current/gripper_width",
               rr.SeriesLines(colors=[[0, 180, 0]], names=["Current"]), static=True)
        rr.log("2D/world/plot/commanded/gripper_width",
               rr.SeriesLines(colors=[[150, 150, 150]], names=["Commanded"]), static=True)
    
    def update_temporal_components(self, **kwargs):
        """Update temporal visualization components. Implementation in visualize() method."""
        pass
    
    def visualize(self, app_id: str = "Robot Episode Viewer", spawn: bool = True):
        """
        Visualize complete episode data with synchronized robot and video streams.

        Args:
            app_id: Rerun application identifier
            spawn: Whether to spawn new Rerun viewer instance
        """
        rr.init(app_id, spawn=spawn)
        self.setup_static_components()

        # Extract data for processing
        data_arrays = self._extract_data_arrays()

        # Normalize timestamps to start from zero
        start_time = data_arrays['ur5_timestamps'][0]
        normalized_timestamps = self._normalize_timestamps(data_arrays, start_time)

        # Optimize frame sampling for performance
        frames = self._calculate_frame_sampling(data_arrays['ur5_poses'])

        # Main visualization loop
        self._run_visualization_loop(frames, data_arrays, normalized_timestamps)

        # Clean up video capture if using MP4
        if data_arrays['video_cap'] is not None:
            data_arrays['video_cap'].release()

        # Keep viewer open
        if spawn:
            import time
            print("\nVisualization complete! Rerun viewer is running.")
            print("Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
    
    def _extract_data_arrays(self) -> Dict[str, Any]:
        """Extract data arrays from loaded episode data."""
        return {
            'ur5_poses': self.data['ur5_poses'],
            'ur5_timestamps': self.data['ur5_timestamps'],
            'ur5_targets': self.data['ur5_targets'],
            'gripper_positions': self.data['gripper_positions'],
            'gripper_timestamps': self.data['gripper_timestamps'],
            'gripper_targets': self.data['gripper_targets'],
            'gripper_target_timestamps': self.data['gripper_target_timestamps'],
            'video_cap': self.data['video_cap'],
            'video_timestamps': self.data['video_timestamps']
        }
    
    def _normalize_timestamps(self, data_arrays: Dict[str, Any], start_time: float) -> Dict[str, np.ndarray]:
        """Normalize all timestamps to start from zero."""
        return {
            'ur5': data_arrays['ur5_timestamps'] - start_time,
            'video': data_arrays['video_timestamps'] - start_time
        }
    
    def _calculate_frame_sampling(self, poses: np.ndarray) -> range:
        """Calculate efficient frame sampling based on max_frames limit."""
        sample_rate = max(1, len(poses) // self.max_frames)
        return range(0, len(poses), sample_rate)
    
    def _run_visualization_loop(self, frames: range, data_arrays: Dict[str, Any],
                               normalized_timestamps: Dict[str, np.ndarray]):
        """Execute main visualization loop for all frames."""
        for i in frames:
            timestamp = normalized_timestamps['ur5'][i]
            rr.set_time("timestamp", timestamp=timestamp)
            
            current_pose = data_arrays['ur5_poses'][i]
            
            # Visualize robot state
            self._visualize_robot_state(current_pose)
            
            # Build and visualize trajectories
            self._update_trajectories(current_pose, data_arrays, i)
            
            # Visualize gripper state
            self._visualize_gripper_state(current_pose, data_arrays, i)
            
            # Update camera visualization
            self._update_camera_visualization(current_pose)
            
            # Process video frames
            self._process_video_frame(i, timestamp, normalized_timestamps['video'], 
                                    data_arrays['video_cap'])
    
    def _visualize_robot_state(self, pose: np.ndarray):
        """Visualize robot pose and coordinate frame."""
        self.log_robot_pose(current_pose=pose, entity_suffix="/tcp")
        self.log_coordinate_frame(pose=pose, scale=0.08, entity_suffix="/axes")
    
    def _update_trajectories(self, current_pose: np.ndarray, data_arrays: Dict[str, Any], frame_idx: int):
        """Update and visualize robot trajectories."""
        # Update actual trajectory
        self.trajectory_positions.append(current_pose[:3])
        if len(self.trajectory_positions) > 1:
            self.log_trajectory(
                positions=np.array(self.trajectory_positions),
                colors=[[0, 255, 0]],
                entity_suffix="/actual_path",
                static=False
            )
        
        # Update target trajectory if available
        ur5_targets = data_arrays['ur5_targets']
        if ur5_targets is not None and frame_idx < len(ur5_targets):
            self.target_trajectory_positions.append(ur5_targets[frame_idx][:3])
            if len(self.target_trajectory_positions) > 1:
                self.log_trajectory(
                    positions=np.array(self.target_trajectory_positions),
                    colors=[[255, 0, 0]],
                    entity_suffix="/target_path",
                    static=False
                )
    
    def _visualize_gripper_state(self, robot_pose: np.ndarray, data_arrays: Dict[str, Any], frame_idx: int):
        """Visualize gripper position and commanded state."""
        # Synchronize gripper data with robot timestamp
        robot_timestamp = data_arrays['ur5_timestamps'][frame_idx]
        gripper_data = self._get_synchronized_gripper_data(robot_timestamp, data_arrays)
        
        # Log gripper data and visualization
        if not np.isnan(gripper_data['current_width']):
            self.log_gripper_data(
                current_width=gripper_data['current_width'],
                commanded_width=gripper_data['commanded_width'],
                entity_suffix=""
            )
            
            self.log_gripper_fingers(
                pose=robot_pose,
                gripper_width=gripper_data['current_width'],
                entity_suffix=""
            )
    
    def _get_synchronized_gripper_data(self, robot_timestamp: float, data_arrays: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Get gripper data synchronized to robot timestamp."""
        # Find closest gripper measurement
        gripper_idx = np.argmin(np.abs(data_arrays['gripper_timestamps'] - robot_timestamp))
        gripper_idx = max(0, min(gripper_idx, len(data_arrays['gripper_positions']) - 1))
        current_width = data_arrays['gripper_positions'][gripper_idx, 0]
        
        # Find closest gripper command if available
        commanded_width = None
        if (data_arrays['gripper_targets'] is not None and 
            data_arrays['gripper_target_timestamps'] is not None):
            target_idx = np.argmin(np.abs(data_arrays['gripper_target_timestamps'] - robot_timestamp))
            target_idx = max(0, min(target_idx, len(data_arrays['gripper_targets']) - 1))
            commanded_width = data_arrays['gripper_targets'][target_idx, 0]
        
        return {
            'current_width': current_width,
            'commanded_width': commanded_width
        }
    
    def _update_camera_visualization(self, gripper_pose: np.ndarray):
        """Update camera pose relative to gripper."""
        self.log_camera_pose_relative_to_gripper(
            gripper_pose=gripper_pose,
            tcp_offset=self.DEFAULT_TCP_OFFSET,
            entity_path="camera/gopro"
        )
    
    def _process_video_frame(self, frame_idx: int, timestamp: float,
                           video_timestamps: np.ndarray, video_cap):
        """Process and log video frame with temporal synchronization."""
        if frame_idx % self.video_rate == 0:
            video_frame_idx = np.argmin(np.abs(video_timestamps - timestamp))
            video_frame_idx = max(0, min(video_frame_idx, self.data['total_frames'] - 1))

            # Handle zarr video (robologger) vs MP4 video (UMI)
            if self.data.get('is_zarr_video', False):
                # Load from zarr array
                frame = np.array(self.data['video_array'][video_frame_idx])
                # Assuming frame is already in RGB format from robologger
                frame_rgb = np.flip(frame, axis=1)  # Flip horizontally for correct view
            else:
                # Load from MP4
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, frame = video_cap.read()

                if not ret or frame is None:
                    return

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 1)  # Flip horizontally for correct view

            rr.log("2D/camera/gopro/image", rr.Image(frame_rgb))
            rr.log("3D/camera/gopro/image", rr.Image(frame_rgb))


# Backwards compatibility aliases
RerunVisualizer = UMIVisualizer


# Backwards compatibility functions
def load_umi_episode_data(episode_path: str) -> Dict[str, Any]:
    """
    Load UMI episode data (backwards compatibility).
    
    Args:
        episode_path: Path to episode directory
        
    Returns:
        Dictionary containing episode data
    """
    viz = UMIVisualizer(episode_path)
    return viz.data


def visualize_umi_episode(episode_path: str, max_frames: int = 500, video_rate: int = 3):
    """
    Visualize UMI episode data (backwards compatibility).
    
    Args:
        episode_path: Path to episode directory
        max_frames: Maximum frames to process
        video_rate: Video sampling rate
    """
    viz = UMIVisualizer(episode_path, max_frames=max_frames, video_rate=video_rate)
    viz.visualize()


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize UMI episode data")
    parser.add_argument("--episode", required=True, 
                       help="Path to UMI episode directory")
    parser.add_argument("--max-frames", type=int, default=500, 
                       help="Maximum frames to process for performance")
    parser.add_argument("--video-rate", type=int, default=3, 
                       help="Video frame sampling rate (process every Nth frame)")
    
    args = parser.parse_args()
    
    visualizer = UMIVisualizer(
        args.episode, 
        max_frames=args.max_frames, 
        video_rate=args.video_rate
    )
    visualizer.visualize()


if __name__ == "__main__":
    main() 
