# Storage data format

project_name/
├── task_name/
    └── run_name/
        └── episode_idx/
            ├── right_arm.zarr/
            │   ├── timestamp
            │   ├── state_pos_xyz
            │   ├── state_quat_wxyz
            │   ├── target_pos_xyz
            │   ├── target_quat_wxyz
            │   └── attr
            ├── right_gripper.zarr/
            │   ├── timestamp
            │   ├── state_joint_pos
            │   ├── target_joint_pos
            │   ├── state_joint_vel (optional)
            │   └── state_joint_torque (optional)
            ├── right_wrist_camera.zarr/
            │   ├── timestamp
            │   ├── camera_name_1.mp4
            │   ├── camera_name_2.mp4
            │   └── attr/
            │       ├── "camera_model": "GoPro Hero 9"
            │       ├── "camera_pose_in_robot_eef_frame": {
            │       │     "pos_xyz": [],
            │       │     "quat_wxyz": []
            │       │   }
            │       └── "camera_config": {
            │             "camera1": {
            │               ...
            │             }
            │           }
            ├── right_force_torque.zarr/
            │   ├── timestamp
            │   ├── torque
            │   └── force
            ├── right_gripper_tactile.zarr/
            ├── left_... (similar structure)
            └── head_... (similar structure)