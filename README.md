# Storage data format

```
project_name/
├── task_name/
│   └── run_name/
│       └── episode_XXXXXX/ # padded to 6 digits
│           ├── metadata.zarr/
│           │   └── .zattrs (project_name, task_name, run_name, morphology, is_demonstration, is_successful)
│           ├── {robot_name}.zarr/ # e.g., right_arm.zarr - cartesian control logger
│           │   ├── state_timestamps
│           │   ├── state_pos_xyz
│           │   ├── state_quat_wxyz
│           │   ├── target_timestamps
│           │   ├── target_pos_xyz
│           │   ├── target_quat_wxyz
│           │   └── .zattrs (ctrl_freq, robot_name)
│           ├── {robot_name}.zarr/ # e.g., right_end_effector.zarr - joint control logger
│           │   ├── state_timestamps
│           │   ├── target_timestamps
│           │   ├── state_joint_pos
│           │   ├── target_joint_pos
│           │   ├── state_joint_vel (optional)
│           │   ├── state_joint_torque (optional)
│           │   └── .zattrs (ctrl_freq, num_joints, robot_name)
│           ├── {camera_system_name}_{idx}.zarr/ # e.g., right_wrist_camera_0.zarr for video logger
│           │   ├── {actual_camera_name}_timestamps # e.g., main_timestamps, depth_timestamps, ultrawide_timestamps
│           │   ├── {actual_camera_name}.mp4         # e.g., main.mp4, depth.mp4, ultrawide.mp4
│           │   └── .zattrs {
│           │       "camera_configs": {
│           │           "camera_name": {
│           │               "fps": 30,
│           │               "height": 720,
│           │               "width": 960,
│           │               "type": "rgb",
│           │               "peertalk_port": 5555
│           │           },
│           │           "depth_camera": {
│           │               "fps": 30,
│           │               "height": 240,
│           │               "width": 320,
│           │               "type": "depth",
│           │               "depth_enc_mode": "hue_codec",
│           │               "depth_range": [0.02, 4.0],
│           │               "peertalk_port": 5557
│           │           }
│           │       }
│           │   }
│           ├── {sensor_name}.zarr/ # future feature - e.g., right_force_torque.zarr
│           │   ├── timestamps
│           │   ├── force
│           │   └── torque
│           ├── {sensor_name}.zarr/ # future feature - e.g., right_gripper_tactile.zarr
│           ├── left_... (similar structure)
│           └── head_... (similar structure)
```
