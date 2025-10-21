# TODO
- Remove opencv dependency (manually write a function to convert between bgr and rgb)
- Enable logging joint velocity and torque in robot control logger
- Add logger to log sensor data (IMU)
- Log language instructions / commands in main logger. Should also include timestamps for each command.
- Depth recording: consider lossless compression?


# Storage data format

```
project_name/
├── task_name/
│   └── run_name/
│       └── episode_XXXXXX/ # padded to 6 digits
│           ├── metadata.zarr/
│           │   └── .zattrs (project_name, task_name, run_name, morphology, is_demonstration, is_successful)
│           ├── {robot_name}.zarr/ # e.g., right_arm.zarr - unified control logger
│           │   ├── state_pos_xyz/
│           │   ├── state_quat_wxyz/
│           │   ├── state_joint_pos/
│           │   ├── state_timestamps/
│           │   ├── target_pos_xyz/ (if target_type=="eef_pose")
│           │   ├── target_quat_wxyz/ (if target_type=="eef_pose")
│           │   ├── target_joint_pos/ (if target_type=="joint_pos")
│           │   ├── target_timestamps/
│           │   └── .zattrs (robot_name, target_type, ctrl_freq, log_eef_pose, log_joint_pos, joint_units, num_joints)
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
