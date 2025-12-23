from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Sequence
import numpy as np
import numpy.typing as npt
import warnings
import rerun as rr
import rerun.blueprint as rrb
from contextlib import contextmanager

from real_env.utils.pose_utils import get_absolute_pose, get_relative_pose


class BaseVisualizer(ABC):
    """
    Base class for cartesian end-effector robot visualizers following rerun documentation best practices.
    """
    
    def __init__(self):
        """Initialize robotics visualizer."""
        pass

    @staticmethod
    def _umi_quat_to_rerun(umi_quat: npt.NDArray[np.float64]) -> List[float]:
        """
        Convert UMI quaternion format to Rerun format.
        
        Args:
            umi_quat: Quaternion in UMI format [qw, qx, qy, qz]
            
        Returns:
            Quaternion in Rerun format [qx, qy, qz, qw]
        """
        return [umi_quat[1], umi_quat[2], umi_quat[3], umi_quat[0]]
    
    @staticmethod
    def _umi_quat_to_scipy(umi_quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert UMI quaternion format to scipy format.
        
        Args:
            umi_quat: Quaternion in UMI format [qw, qx, qy, qz]
            
        Returns:
            Quaternion in scipy format [qx, qy, qz, qw]
        """
        return np.array([umi_quat[1], umi_quat[2], umi_quat[3], umi_quat[0]])
    
    @staticmethod
    def _create_rotation_from_umi_quat(umi_quat: npt.NDArray[np.float64]):
        """
        Create scipy Rotation object from UMI quaternion format.
        
        Args:
            umi_quat: Quaternion in UMI format [qw, qx, qy, qz]
            
        Returns:
            scipy.spatial.transform.Rotation object
        """
        from scipy.spatial.transform import Rotation
        scipy_quat = BaseVisualizer._umi_quat_to_scipy(umi_quat)
        return Rotation.from_quat(scipy_quat)
    
    @staticmethod
    def _transform_offset_to_world(position: npt.NDArray[np.float64], 
                                  umi_quat: npt.NDArray[np.float64], 
                                  local_offset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform local offset to world coordinates using pose.
        
        Args:
            position: World position [x, y, z]
            umi_quat: Quaternion in UMI format [qw, qx, qy, qz]
            local_offset: Offset in local coordinate frame [x, y, z]
            
        Returns:
            World position after applying offset
        """
        rotation = BaseVisualizer._create_rotation_from_umi_quat(umi_quat)
        world_offset = rotation.apply(local_offset)
        return position + world_offset

    def log_robot_pose(self, current_pose: npt.NDArray[np.float64], commanded_pose: Optional[npt.NDArray[np.float64]] = None, 
                       entity_suffix: str = ""):
        """
        Log temporal robot pose data (current vs commanded).
        
        Robot poses are always temporal - they represent the robot's state over time.
        
        Args:
            current_pose: [x, y, z, qw, qx, qy, qz] current robot pose (from sensors)        
            commanded_pose: [x, y, z, qw, qx, qy, qz] commanded robot pose (from trajectory planner)
            entity_suffix: Additional path suffix
        """
        try:
            current_quat_rerun = self._umi_quat_to_rerun(current_pose[3:7])
            
            rr.log(
                f"3D/world/current/pose{entity_suffix}",
                rr.Transform3D(
                    translation=current_pose[:3],
                    rotation=rr.Quaternion(xyzw=current_quat_rerun)
                )
            )
            
            if commanded_pose is not None:
                commanded_quat_rerun = self._umi_quat_to_rerun(commanded_pose[3:7])
                rr.log(
                    f"3D/world/commanded/pose{entity_suffix}",
                    rr.Transform3D(
                        translation=commanded_pose[:3],
                        rotation=rr.Quaternion(xyzw=commanded_quat_rerun)
                    )
                )
                
        except Exception as e:
            warnings.warn(f"Failed to log robot pose: {e}")
    
    def log_gripper_data(self, current_width: float, commanded_width: Optional[float] = None, entity_suffix: str = ""):
        """
        Log temporal gripper data (current vs commanded).

        Args:
            current_width: Current gripper opening width (from sensors)
            commanded_width: Commanded gripper opening width (from controller)
            entity_suffix: Additional path suffix
        """
        try:
            rr.log(
                f"2D/world/data/current/gripper_width{entity_suffix}",
                rr.Scalars([current_width])
            )
            rr.log(
                f"2D/world/plot/current/gripper_width{entity_suffix}",
                rr.Scalars([current_width])
            )

            if commanded_width is not None:
                rr.log(
                    f"2D/world/data/commanded/gripper_width{entity_suffix}",
                    rr.Scalars([commanded_width])
                )
                rr.log(
                    f"2D/world/plot/commanded/gripper_width{entity_suffix}",
                    rr.Scalars([commanded_width])
                )

        except Exception as e:
            print(f"Warning: Failed to log gripper data: {e}")

    def log_gripper_fingers(self, pose: npt.NDArray[np.float64], gripper_width: float,
                           entity_suffix: str = "/fingers"):
        """
        Log gripper fingers visualization to show opening/closing.
        
        Args:
            pose: Gripper pose [x, y, z, qw, qx, qy, qz] - UMI format
            gripper_width: Current gripper width in meters
            entity_suffix: Additional path suffix (default: "/fingers")
        """
        try:
            pos = pose[:3]
            gripper_quat_umi = pose[3:7]
            
            # Define finger offsets in gripper's local coordinate frame
            half_width = gripper_width / 2.0
            left_offset = np.array([-half_width, 0.0, 0.0])
            right_offset = np.array([half_width, 0.0, 0.0])
            
            left_finger_pos = self._transform_offset_to_world(pos, gripper_quat_umi, left_offset)
            right_finger_pos = self._transform_offset_to_world(pos, gripper_quat_umi, right_offset)
            
            quat_rerun = self._umi_quat_to_rerun(gripper_quat_umi)
            
            rr.log(
                f"3D/world/current/gripper_left{entity_suffix}",
                rr.Transform3D(
                    translation=left_finger_pos,
                    rotation=rr.Quaternion(xyzw=quat_rerun)
                )
            )
            
            rr.log(
                f"3D/world/current/gripper_right{entity_suffix}",
                rr.Transform3D(
                    translation=right_finger_pos,
                    rotation=rr.Quaternion(xyzw=quat_rerun)
                )
            )
            
            finger_size = [0.005, 0.01, 0.015]  
            rr.log(
                f"3D/world/current/gripper_left{entity_suffix}/box",
                rr.Boxes3D(
                    sizes=[finger_size],
                    colors=[[100, 100, 255]]
                )
            )
            rr.log(
                f"3D/world/current/gripper_right{entity_suffix}/box",
                rr.Boxes3D(
                    sizes=[finger_size],
                    colors=[[100, 100, 255]]
                )
            )
            
        except Exception as e:
            warnings.warn(f"Failed to log gripper fingers: {e}")

    def log_camera_pose_relative_to_gripper(self, gripper_pose: npt.NDArray[np.float64], 
                                           tcp_offset: float = 0.205,
                                           entity_path: str = "camera/gopro"):
        """
        Log camera pose relative to gripper TCP using UMI camera mounting configuration.
        
        Camera mounting for UMI setup with GoPro Hero 9/10/11:
        - Y-axis offset: 0.086m (cam_to_center_height)
        - Z-axis offset: 0.01465m (optical center to mount) + tcp_offset
        
        Args:
            gripper_pose: Gripper pose [x, y, z, qw, qx, qy, qz] - UMI format
            tcp_offset: TCP offset in meters (default: 0.13 for UMI)
            entity_path: Camera entity path (without 3D/2D prefix)
        """
        try:
            # UMI camera mounting constants (all units in meters)
            cam_to_center_height = 0.086
            cam_to_mount_offset = 0.01465
            cam_to_tip_offset = cam_to_mount_offset + tcp_offset
            
            _, gripper_quat = gripper_pose[:3], gripper_pose[3:7]
            
            # Camera offset in TCP coordinate frame (UMI mounting configuration)
            cam_in_gripper_frame = get_absolute_pose(gripper_pose, 
                                                     np.array([0.0, -cam_to_center_height, -cam_to_tip_offset, 1.0, 0.0, 0.0, 0.0]))
                
            camera_pos = cam_in_gripper_frame[:3]
            camera_quat_rerun = self._umi_quat_to_rerun(gripper_quat)
            
            rr.log(
                f"3D/{entity_path}",
                rr.Transform3D(
                    translation=camera_pos,
                    rotation=rr.Quaternion(xyzw=camera_quat_rerun)
                )
            )
            
        except Exception as e:
            warnings.warn(f"Failed to log camera pose relative to gripper: {e}")

    def log_video_frame(self, image: npt.NDArray[np.uint8], timestamp: float, 
                       entity_suffix: str = "/image"):
        """
        Log temporal video frame synchronized to timestamp.
        
        Args:
            image: Image array (H, W, C) in RGB format, shape (height, width, 3)
            timestamp: Frame timestamp in seconds (for synchronization)
            entity_suffix: Additional path suffix
        """
        try:
            rr.log(
                f"2D/world{entity_suffix}",
                rr.Image(image)
            )
            
        except Exception as e:
            warnings.warn(f"Failed to log video frame: {e}")
    
    def log_trajectory(self, positions: npt.NDArray[np.float64], colors: Optional[List[List[int]]] = None,
                      entity_suffix: str = "/trajectory", static: bool = False):
        """
        Log robot trajectory as line strips.
        
        Two use cases:
        - static=False (default): Progressive trajectory that builds over time (temporal)
        - static=True: Complete trajectory shown as reference (e.g., entire recorded path)
        
        Args:
            positions: Array of 3D positions, shape (N, 3) where N is number of points
            colors: List of RGB colors for trajectory segments, each color as [R, G, B]
            entity_suffix: Additional path suffix
            static: True for complete reference trajectory, False for progressive temporal trajectory
        """
        try:
            if colors is None:  
                colors = [[100, 150, 255]]  
                
            rr.log(
                f"3D/world{entity_suffix}",
                rr.LineStrips3D(
                    strips=[positions],
                    colors=colors,
                    radii=0.001
                ),
                static=static
            )
            
        except Exception as e:
            warnings.warn(f"Failed to log trajectory: {e}")
    
    def log_coordinate_frame(self, pose: npt.NDArray[np.float64], scale: float = 0.1,
                           entity_suffix: str = "/coordinate_frame", static: bool = False):
        """
        Log coordinate frame arrows at given pose.
        
        For robot end-effector/gripper poses:
        - Z-axis (BLUE) points in the gripper's forward direction (approach vector)
        - X-axis (RED) points along gripper opening direction  
        - Y-axis (GREEN) completes the right-handed coordinate frame
        
        Args:
            pose: [x, y, z, qw, qx, qy, qz] robot pose where Z points forward
            scale: Length of coordinate arrows (meters)
            entity_suffix: Additional path suffix
            static: Whether this is static data
        """
        try:
            position = pose[:3]
            umi_quat = pose[3:7]
            
            rotation = self._create_rotation_from_umi_quat(umi_quat)
            rotation_matrix = rotation.as_matrix()
            
            # Create coordinate frame vectors for gripper convention
            origins = [position] * 3
            x_vec = rotation_matrix @ np.array([scale, 0, 0])
            y_vec = rotation_matrix @ np.array([0, scale, 0])
            z_vec = rotation_matrix @ np.array([0, 0, scale])
            
            vectors = [x_vec, y_vec, z_vec]
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
            
            rr.log(
                f"3D/world{entity_suffix}",
                rr.Arrows3D(
                    origins=origins,
                    vectors=vectors,
                    colors=colors,
                    labels=["X (Opening)", "Y (Side)", "Z (Forward)"]
                ),
                static=static
            )
            
        except Exception as e:
            warnings.warn(f"Failed to log coordinate frame: {e}")
    
    @staticmethod
    def create_robotics_blueprint(camera_names: Optional[List[str]] = None) -> rrb.Blueprint:
        """
        Create standard robotics visualization blueprint.
        
        Layout:
        - Top Left (70%): 3D Robot Scene
        - Top Right (30%): Camera views (max 3 cameras vertical)
        - Bottom: Combined gripper plot + additional cameras if needed
        
        Camera Layout Logic:
        - ≤3 cameras: All in top right (30%)
        - 4-5 cameras: 3 in top right, 2 in bottom row next to gripper plot
        - ≥6 cameras: 3 in top right, 2 in bottom row, rest in tabs
        
        Args:
            camera_names: List of camera names for video views
            
        Returns:
            Blueprint for robotics visualization layout
        """
        assert camera_names is not None, "Camera names must be provided"
        
        num_cameras = len(camera_names)
        
        camera_views = []
        for cam_name in camera_names:
            camera_views.append(
                rrb.Spatial2DView(
                    origin=f"2D/camera/{cam_name}/image",
                    name=f"{cam_name} Camera"
                )
            )
        
        combined_gripper_plot = rrb.TimeSeriesView(
            origin="2D/world/plot",
            name="Gripper Width (Current vs Commanded)"
        )
        
        main_3d_view = rrb.Spatial3DView(
            origin="3D",
            name="3D Robot Scene",
            contents=[
                "+ 3D/**",
            ]
        )
        
        # Camera layout logic
        if num_cameras <= 3:
            top_right = rrb.Vertical(*camera_views)
            bottom_row = combined_gripper_plot
            
        elif num_cameras <= 5:
            top_cameras = camera_views[:3]
            bottom_cameras = camera_views[3:5]
            
            top_right = rrb.Vertical(*top_cameras)
            bottom_row = rrb.Horizontal(
                combined_gripper_plot,
                rrb.Horizontal(*bottom_cameras),
                column_shares=[40, 60]
            )
            
        else:
            top_cameras = camera_views[:3]
            bottom_cameras = camera_views[3:5]
            extra_cameras = camera_views[5:]
            
            top_right = rrb.Vertical(*top_cameras)
            
            bottom_camera_section = rrb.Horizontal(
                rrb.Horizontal(*bottom_cameras),
                rrb.Tabs(*extra_cameras),
                column_shares=[50, 50]
            )
            
            bottom_row = rrb.Horizontal(
                combined_gripper_plot,
                bottom_camera_section,
                column_shares=[50, 50]
            )
        
        top_row = rrb.Horizontal(
            main_3d_view,
            top_right,
            column_shares=[70, 30]
        )
        
        return rrb.Vertical(
            top_row,
            bottom_row,
            row_shares=[70, 30]
        )
    
    def setup_world_coordinates(self, coordinate_system: rr.ViewCoordinates = rr.ViewCoordinates.RIGHT_HAND_Z_UP):
        """Setup world coordinate system (typically called once)."""
        try:
            rr.log("world", coordinate_system, static=True)
        except Exception as e:
            warnings.warn(f"Failed to setup world coordinates: {e}") 
