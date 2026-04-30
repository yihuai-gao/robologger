<!--
 Copyright (c) 2024 Yihuai Gao

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# robologger

## TL;DR

`robologger` is a distributed data collection library for robot learning. Each modality (robot arm, camera, gripper, mobile base) runs its own logger process at its own frequency; a central `MainLogger` coordinates episode start/stop via [robotmq](https://github.com/yihuai-gao/robotmq). Numerical data is stored in zarr; video is encoded to MP4 via FFmpeg.

**Install:**
```bash
pip install robologger
```

**Key features:**
- Distributed multi-process design — each control loop logs independently at its own rate, no blocking
- In-memory buffering during recording, single batch zarr write on stop (low-latency logging)
- Structured zarr episode storage with standardized naming conventions
- Video encoding via FFmpeg (GPU-accelerated h264_nvenc by default) with hue-codec depth support
- Automatic episode management: directory creation, metadata, delete-last-episode
- Configurable success labeling (prompt or hardcoded)

**Common use cases:**
- **Robot arm control loop**: log joint positions, EEF pose, and control targets at 100–125 Hz
- **Camera loop**: log RGB and depth video streams from multiple cameras
- **Gripper / end-effector loop**: log gripper width and commands
- **Mobile base**: log 2D pose and velocity
- **Multi-robot / bi-manual**: run one logger per body part; one `MainLogger` coordinates all

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
  - [Multi-Process Design](#multi-process-design)
  - [Communication via RobotMQ](#communication-via-robotmq)
  - [In-Memory Buffering](#in-memory-buffering)
- [Data Storage Format](#data-storage-format)
- [Naming Conventions](#naming-conventions)
  - [Morphology](#morphology)
  - [RobotName](#robotname)
  - [CameraName](#cameraname)
- [Logger Classes](#logger-classes)
  - [MainLogger](#mainlogger)
  - [RobotCtrlLogger](#robotctrllogger)
  - [VideoLogger](#videologger)
  - [MobileBaseLogger](#mobilebaselogger)
- [Usage Examples](#usage-examples)
  - [Running the Full Example System](#running-the-full-example-system)
  - [Joint-Controlled Robot Arm](#joint-controlled-robot-arm)
  - [EEF-Controlled Robot Arm](#eef-controlled-robot-arm)
  - [Camera Loop (RGB + Depth)](#camera-loop-rgb--depth)
  - [Gripper / End Effector](#gripper--end-effector)
  - [Mobile Base](#mobile-base)
  - [Main Coordinator Process](#main-coordinator-process)
- [Post-Processing Scripts](#post-processing-scripts)

---

## Overview

`robologger` solves the data collection problem for robot learning experiments. In a typical robot system, multiple processes run concurrently at different frequencies: a camera loop at 30 Hz, a robot controller at 100–125 Hz, a gripper at 30 Hz. Each process needs to:

1. Run continuously regardless of whether data collection is active
2. Start and stop logging on demand from a central operator
3. Save data efficiently without disrupting real-time control

`robologger` provides a logger class for each modality, all controlled by a single `MainLogger` via [robotmq](https://github.com/yihuai-gao/robotmq) message passing. Data is saved in a standardized zarr directory structure with episode-level metadata.

---



## Installation

### From PyPI (recommended)

```bash
pip install robologger
```

### From Source

```bash
git clone https://github.com/yihuai-gao/robologger
cd robologger
pip install -e .
```

**Dependencies:** `numpy`, `zarr>=2.16.0,<3.0.0`, `loguru`, `robotmq`, `ffmpeg` (system package for video encoding).

**Optional: GPU-accelerated video encoding**

The default codec is `h264_nvenc` (NVIDIA GPU). If you don't have an NVIDIA GPU, pass `codec="libx264"` to `VideoLogger`:

```python
VideoLogger(..., codec="libx264")
```

---

## Architecture

### Multi-Process Design

```
┌─────────────────────────────────────────────────────────┐
│  Main Process  (operator-facing; keyboard commands)      │
│  MainLogger                                              │
│   ├── RMQClient → "right_wrist_camera_0" endpoint       │
│   ├── RMQClient → "right_arm" endpoint                   │
│   └── RMQClient → "right_end_effector" endpoint          │
└──────────┬──────────────────────────────────────────────┘
           │  start / stop / pause / resume (via RMQ)
    ┌──────┴─────────────────────────────────────────────┐
    │                                                     │
    ▼                   ▼                    ▼
┌──────────┐   ┌──────────────┐   ┌─────────────────────┐
│ Camera   │   │ Robot Ctrl   │   │ Gripper Process     │
│ Process  │   │ Process      │   │                     │
│ 30 Hz    │   │ 100-125 Hz   │   │ 30 Hz               │
│          │   │              │   │                     │
│VideoLogger│  │RobotCtrlLogger│  │ RobotCtrlLogger     │
│ RMQServer│   │ RMQServer    │   │ RMQServer           │
└──────────┘   └──────────────┘   └─────────────────────┘
```

Each control loop is a separate OS process. Control loops run continuously and only write data when recording is active. The `MainLogger` in the main process discovers loggers, sends start/stop commands, and manages episode directories.

### Communication via RobotMQ

Every `BaseLogger` subclass internally starts an `RMQServer` with two topics:
- `"command"`: receives start / stop / pause / resume commands from `MainLogger`
- `"info"`: exposes `{"name": ..., "attr": ...}` so `MainLogger` can validate it connected to the right logger

`MainLogger` holds an `RMQClient` per logger endpoint and calls `put_data("command", ...)` to control recording.

### In-Memory Buffering

All numerical data (joint positions, EEF pose, timestamps) is appended to Python lists in RAM during recording. When `stop_recording()` is called, the entire buffer is written to zarr in a single batch. This keeps the logging hot-path allocation-free — no zarr resize overhead per step.

Video frames are a special case: they are piped directly to a per-camera FFmpeg process as raw bytes, which encodes them to MP4 in real time. Timestamps are buffered in memory and written to zarr at stop.

---

## Data Storage Format

```
{root_dir}/
└── {project_name}/
    └── {task_name}/
        └── {run_name}/
            └── episode_XXXXXX/       # zero-padded 6-digit index
                ├── metadata.zarr/
                │   └── .zattrs       # project_name, task_name, run_name,
                │                     # morphology, is_demonstration, is_successful
                │
                ├── {robot_name}.zarr/           # e.g., right_arm.zarr
                │   ├── state_timestamps          # (N,) float64
                │   ├── state_pos_xyz             # (N, 3) float64 — if log_eef_pose=True
                │   ├── state_quat_wxyz           # (N, 4) float64 — if log_eef_pose=True
                │   ├── state_joint_pos           # (N, J) float64 — if log_joint_pos=True
                │   ├── state_joint_vel           # (N, J) float64 — if provided
                │   ├── state_joint_torque        # (N, J) float64 — if provided
                │   ├── target_timestamps         # (M,) float64
                │   ├── target_pos_xyz            # (M, 3) float64 — if target_type="eef_pose"
                │   ├── target_quat_wxyz          # (M, 4) float64 — if target_type="eef_pose"
                │   ├── target_joint_pos          # (M, J) float64 — if target_type="joint_pos"
                │   └── .zattrs       # target_type, log_eef_pose, log_joint_pos,
                │                     # joint_units, num_joints, + any extra attr keys
                │
                ├── {camera_system_name}_{idx}.zarr/   # e.g., right_wrist_camera_0.zarr
                │   ├── {camera_name}.mp4              # e.g., main.mp4, depth.mp4
                │   ├── {camera_name}_timestamps       # (F,) float64 per camera
                │   └── .zattrs       # camera_configs dict
                │
                └── {mobile_base_name}.zarr/
                    ├── state_timestamps
                    ├── state_pose            # (N, 3) [pos_x, pos_y, rot_yaw]
                    ├── state_velocity        # (N, 3) [vel_x, vel_y, vel_yaw]
                    ├── target_timestamps
                    ├── target_pose
                    └── target_velocity
```

State and target can have different timestamps and different numbers of samples (N ≠ M). This is expected since sensing and commanding may run at different rates or triggered by different events.

---

## Naming Conventions

All logger names are validated against enums in `robologger/classes.py`.

### Morphology

Used in episode metadata to describe the robot body type.

| Value | Enum |
|---|---|
| `"single_arm"` | `Morphology.SINGLE_ARM` |
| `"bi_manual"` | `Morphology.BI_MANUAL` |
| `"wheel_based_single_arm"` | `Morphology.WHEEL_BASED_SINGLE_ARM` |
| `"wheel_based_bi_manual"` | `Morphology.WHEEL_BASED_BI_MANUAL` |
| `"humanoid"` | `Morphology.HUMANOID` |

### RobotName

Required for `RobotCtrlLogger` names.

| Value | Usage |
|---|---|
| `"right_arm"` | Right robot arm |
| `"left_arm"` | Left robot arm |
| `"head"` | Head / neck |
| `"body"` | Torso |
| `"right_end_effector"` | Right gripper or hand |
| `"left_end_effector"` | Left gripper or hand |

### CameraName

`VideoLogger` names must match `{base}{index}` where the base is one of:

| Base | Example names |
|---|---|
| `"right_wrist_camera_"` | `right_wrist_camera_0`, `right_wrist_camera_1` |
| `"left_wrist_camera_"` | `left_wrist_camera_0` |
| `"head_camera_"` | `head_camera_0` |
| `"body_camera_"` | `body_camera_0` |
| `"third_person_camera_"` | `third_person_camera_0` |

Multiple loggers can share the same base (e.g., three iPhones on the right wrist → `right_wrist_camera_0`, `right_wrist_camera_1`, `right_wrist_camera_2`).

---

## Logger Classes

### MainLogger

Coordinates recording across all sub-loggers. Runs in the operator's main process.

```python
from robologger import MainLogger
from robologger.classes import Morphology

main_logger = MainLogger(
    name="main_logger",
    root_dir="data",                      # Episode data root; relative paths resolved to cwd
    project_name="my_project",
    task_name="pick_and_place",
    run_name="run_001",
    logger_endpoints={                    # {logger_name: RMQ endpoint}
        "right_wrist_camera_0": "tcp://localhost:55555",
        "right_arm":            "tcp://localhost:55556",
        "right_end_effector":   "tcp://localhost:55557",
    },
    morphology=Morphology.SINGLE_ARM,
    is_demonstration=True,
    success_config="input_true",          # See table below
)
```

**`success_config` options:**

| Value | Behavior |
|---|---|
| `"none"` | `is_successful` field not set in metadata |
| `"input_true"` | Prompt `[y]/n` on stop — defaults to successful |
| `"input_false"` | Prompt `y/[n]` on stop — defaults to failed |
| `"hardcode_true"` | Always marks successful, no prompt |
| `"hardcode_false"` | Always marks failed, no prompt |

#### Methods

```python
episode_idx = main_logger.start_recording(
    episode_idx=None,           # auto-increment if None
    episode_config={"lang": "pick up the cup"}  # optional dict saved in metadata.zarr/.zattrs
)
```
Validates all logger endpoints are alive, creates the episode directory, writes episode config to `metadata.zarr`, and sends `start` commands to all loggers.

```python
episode_idx = main_logger.stop_recording(
    is_successful=None          # overrides success_config if provided
)
```
Sends `stop` to all alive loggers, stores episode metadata (`project_name`, `task_name`, `run_name`, `morphology`, `is_demonstration`, `is_successful`), and returns the episode index.

```python
main_logger.delete_last_episode()   # Prompts for confirmation; deletes the episode directory
main_logger.delete_episode(idx)     # Delete a specific episode by index (no prompt)

alive = main_logger.get_alive_loggers()  # Returns list of logger names that are reachable
```

---

### RobotCtrlLogger

Logs robot arm or end-effector control data. Data is buffered in memory and written to zarr at `stop_recording()`.

```python
from robologger import RobotCtrlLogger

logger = RobotCtrlLogger(
    name="right_arm",                   # Must match RobotName enum
    endpoint="tcp://localhost:55556",   # RMQ server endpoint
    attr={
        "num_joints": 7,                # Required when log_joint_pos=True
        # Any extra keys (e.g. ctrl_freq) are saved as zarr attributes
    },
    log_eef_pose=True,                  # Log xyz position + wxyz quaternion
    log_joint_pos=True,                 # Log joint position vector
    target_type="joint_pos",            # "joint_pos" or "eef_pose"
    joint_units="radians",              # "radians", "meters", or None
)
```

**Configuration constraints:**
- At least one of `log_eef_pose` or `log_joint_pos` must be `True`
- `target_type="eef_pose"` requires `log_eef_pose=True`
- `target_type="joint_pos"` requires `log_joint_pos=True`
- `joint_units` required (not `None`) when `log_joint_pos=True`
- `attr["num_joints"]` required when `log_joint_pos=True`

#### Control Loop Integration

```python
while True:
    # update_recording_state() polls the RMQ command topic.
    # Returns True if recording is active.
    if logger.update_recording_state():
        logger.log_state(
            state_timestamp=time.monotonic(),
            state_pos_xyz=pos,           # (3,) float64 — required if log_eef_pose=True
            state_quat_wxyz=quat,        # (4,) float64 wxyz — required if log_eef_pose=True
            state_joint_pos=joints,      # (J,) float64 — required if log_joint_pos=True
            state_joint_vel=vel,         # (J,) float64 — optional
            state_joint_torque=torque,   # (J,) float64 — optional
        )
        logger.log_target(
            target_timestamp=time.monotonic(),
            target_joint_pos=target,     # (J,) float64 — required if target_type="joint_pos"
            # target_pos_xyz=...,        # required if target_type="eef_pose"
            # target_quat_wxyz=...,      # required if target_type="eef_pose"
        )
```

`state_joint_vel` and `state_joint_torque` are lazily initialized: the zarr datasets are only created if you ever pass them to `log_state()`. Target joint vel/torque are not supported.

---

### VideoLogger

Logs multi-camera video streams. Each `VideoLogger` instance manages one camera subsystem (e.g., one iPhone with 3 lenses). Frames are piped to FFmpeg for real-time MP4 encoding.

```python
from robologger import VideoLogger

logger = VideoLogger(
    name="right_wrist_camera_0",        # Must match CameraName enum pattern
    endpoint="tcp://localhost:55555",
    attr={
        "camera_configs": {
            "main": {
                "type": "rgb",          # "rgb" or "depth"
                "width": 960,
                "height": 720,
                "fps": 30,
            },
            "depth": {
                "type": "depth",        # float HW array, encoded via hue codec
                "width": 320,
                "height": 240,
                "fps": 30,
            },
        }
    },
    codec="h264_nvenc",                 # FFmpeg video codec (default: GPU-accelerated h264)
    depth_range=(0.0, 4.0),            # Depth encoding range in meters
)
```

**Camera config required keys:** `type`, `width`, `height`, `fps`.

#### Control Loop Integration

```python
while True:
    if logger.update_recording_state():
        # Log a single camera
        logger.log_frame(
            camera_name="main",
            timestamp=time.monotonic(),
            frame=rgb_array,            # (H, W, 3) uint8 for rgb; (H, W) float16/float32/float64 for depth
        )

        # Or log multiple cameras at once
        logger.log_frames({
            "main":  {"frame": rgb_array,   "timestamp": t},
            "depth": {"frame": depth_array, "timestamp": t},
        })
```

**Depth encoding:** Depth frames (float16/float32/float64, in meters) are converted to RGB using a hue-based encoding before passing to FFmpeg. Use `VideoLogger.depth_range` to set the min/max depth in meters that maps to the full hue range.

---

### MobileBaseLogger

Logs 2D mobile base pose and velocity (x, y, yaw).

```python
from robologger.loggers.mobile_base_logger import MobileBaseLogger

logger = MobileBaseLogger(
    name="body",                        # Must match RobotName enum
    endpoint="tcp://localhost:55560",
    attr={},
    log_state_pose=True,                # Log (pos_x, pos_y, rot_yaw)
    log_state_velocity=True,            # Log (vel_x, vel_y, vel_yaw)
    log_target_pose=False,
    log_target_velocity=True,
)
```

At least one of `log_state_pose`/`log_state_velocity` must be `True`, and similarly for target.

```python
while True:
    t = time.monotonic()
    if logger.update_recording_state():
        logger.log_state(
            state_timestamp=t,
            state_pose=np.array([x, y, yaw]),
            state_velocity=np.array([vx, vy, vyaw]),
        )
        logger.log_target(
            target_timestamp=t,
            target_velocity=np.array([cmd_vx, cmd_vy, cmd_vyaw]),
        )
```

---

## Usage Examples

### Running the Full Example System

```bash
# Terminal 1: Camera loop (RGB + depth from simulated iPhone)
python examples/camera_loop.py

# Terminal 2: Robot arm control loop
python examples/robot_control_loop.py --mode joint      # recommended
python examples/robot_control_loop.py --mode cartesian

# Terminal 3: Gripper loop
python examples/gripper_control_loop.py

# Terminal 4: Main coordinator (keyboard-driven)
python examples/main_process.py
```

**Keyboard commands in main process:**

| Key | Action |
|---|---|
| `s` | Start recording episode |
| `e` | End recording episode |
| `d` | Delete last completed episode (with confirmation) |
| `q` / Ctrl+C | Quit |

### Joint-Controlled Robot Arm

The recommended setup for most robot arms (UR, Franka, Kinova, etc.): control via joint positions, log both joint positions and EEF pose (from forward kinematics) for maximum information richness.

```python
import time
import numpy as np
from robologger import RobotCtrlLogger

logger = RobotCtrlLogger(
    name="right_arm",
    endpoint="tcp://localhost:55556",
    attr={"num_joints": 7},
    log_eef_pose=True,
    log_joint_pos=True,
    target_type="joint_pos",
    joint_units="radians",
)

while True:
    loop_start = time.monotonic()
    t = time.monotonic()

    if logger.update_recording_state():
        logger.log_state(
            state_timestamp=t,
            state_joint_pos=robot.get_joint_pos(),            # (7,) float64
            state_pos_xyz=robot.get_eef_pos(),                # (3,) float64
            state_quat_wxyz=robot.get_eef_quat_wxyz(),        # (4,) float64
        )
        logger.log_target(
            target_timestamp=t,
            target_joint_pos=controller.get_target(),         # (7,) float64
        )

    time.sleep(max(0, 0.01 - (time.monotonic() - loop_start)))
```

### EEF-Controlled Robot Arm

For Cartesian-space controllers. If the robot has joint encoders, log both EEF pose and joint positions.

```python
logger = RobotCtrlLogger(
    name="right_arm",
    endpoint="tcp://localhost:55556",
    attr={"num_joints": 7},
    log_eef_pose=True,
    log_joint_pos=True,
    target_type="eef_pose",
    joint_units="radians",
)

while True:
    t = time.monotonic()
    if logger.update_recording_state():
        logger.log_state(
            state_timestamp=t,
            state_pos_xyz=robot.get_eef_pos(),
            state_quat_wxyz=robot.get_eef_quat_wxyz(),
            state_joint_pos=robot.get_joint_pos(),
        )
        logger.log_target(
            target_timestamp=t,
            target_pos_xyz=target_pos,
            target_quat_wxyz=target_quat,
        )
    time.sleep(1/125)
```

### Camera Loop (RGB + Depth)

```python
from robologger import VideoLogger
import numpy as np, time

logger = VideoLogger(
    name="right_wrist_camera_0",
    endpoint="tcp://localhost:55555",
    attr={"camera_configs": {
        "main":  {"type": "rgb",   "width": 960, "height": 720, "fps": 30},
        "depth": {"type": "depth", "width": 320, "height": 240, "fps": 30},
    }},
)

while True:
    t = time.monotonic()
    if logger.update_recording_state():
        # Replace with real camera capture:
        # rgb = camera.get_rgb()     # (720, 960, 3) uint8
        # depth = camera.get_depth() # (240, 320) float32, values in meters
        rgb = np.zeros((720, 960, 3), dtype=np.uint8)
        depth = np.zeros((240, 320), dtype=np.float32)

        logger.log_frames({
            "main":  {"frame": rgb,   "timestamp": t},
            "depth": {"frame": depth, "timestamp": t},
        })
    time.sleep(1/30)
```

### Gripper / End Effector

```python
logger = RobotCtrlLogger(
    name="right_end_effector",
    endpoint="tcp://localhost:55557",
    attr={"num_joints": 1},
    log_eef_pose=False,
    log_joint_pos=True,
    target_type="joint_pos",
    joint_units="meters",
)

while True:
    t = time.monotonic()
    if logger.update_recording_state():
        logger.log_state(state_timestamp=t, state_joint_pos=np.array([gripper.get_width()]))
        logger.log_target(target_timestamp=t, target_joint_pos=np.array([target_width]))
    time.sleep(1/30)
```

### Mobile Base

```python
from robologger.loggers.mobile_base_logger import MobileBaseLogger

logger = MobileBaseLogger(
    name="body",
    endpoint="tcp://localhost:55560",
    attr={},
    log_state_pose=True,
    log_state_velocity=True,
    log_target_velocity=True,
)

while True:
    t = time.monotonic()
    if logger.update_recording_state():
        logger.log_state(state_timestamp=t, state_pose=np.array([x, y, yaw]), state_velocity=np.array([vx, vy, vyaw]))
        logger.log_target(target_timestamp=t, target_velocity=np.array([cmd_vx, cmd_vy, cmd_vyaw]))
    time.sleep(1/30)
```

### Main Coordinator Process

```python
from robologger import MainLogger
from robologger.classes import Morphology

main_logger = MainLogger(
    name="main_logger",
    root_dir="data",
    project_name="my_project",
    task_name="pick_and_place",
    run_name="run_001",
    logger_endpoints={
        "right_wrist_camera_0": "tcp://localhost:55555",
        "right_arm":            "tcp://localhost:55556",
        "right_end_effector":   "tcp://localhost:55557",
    },
    morphology=Morphology.SINGLE_ARM,
    is_demonstration=True,
    success_config="input_true",
)

# Start an episode (auto-increments index; optionally provide episode_config)
episode_idx = main_logger.start_recording(
    episode_config={"instruction": "pick up the red cup"}
)

# ... wait for episode to complete ...

episode_idx = main_logger.stop_recording()   # prompts for success if success_config="input_true"

# Delete the last episode if it was bad (with interactive confirmation)
main_logger.delete_last_episode()
```

**Reading saved data:**

```python
import zarr, numpy as np

# Episode metadata
meta = zarr.open_group("data/my_project/pick_and_place/run_001/episode_000000/metadata.zarr", "r")
print(meta.attrs.asdict())
# {'project_name': 'my_project', 'task_name': 'pick_and_place', ..., 'is_successful': True}

# Robot arm data
arm = zarr.open_group("data/my_project/pick_and_place/run_001/episode_000000/right_arm.zarr", "r")
joint_pos = np.array(arm["state_joint_pos"])   # (N, 7)
timestamps = np.array(arm["state_timestamps"]) # (N,)

# Camera timestamps
cam = zarr.open_group(".../right_wrist_camera_0.zarr", "r")
frame_times = np.array(cam["main_timestamps"])  # (F,)
# Video: .../right_wrist_camera_0.zarr/main.mp4
```

---

## Post-Processing Scripts

### aggregate_data

Aggregates video files from all episodes in a run directory into a flat folder, optionally separated by success status.

```bash
# Aggregate all videos from a run
aggregate_data /path/to/run_dir

# Aggregate only the "main" camera stream
aggregate_data /path/to/run_dir --camera_name main

# Custom output folder name
aggregate_data /path/to/run_dir --output_name collected_videos

# Separate successful/failed episodes into sub-folders
aggregate_data /path/to/run_dir --separate_successful_episodes

# Use more parallel workers (default: min(4, cpu_count))
aggregate_data /path/to/run_dir --max_workers 12
```

Output structure (without `--separate_successful_episodes`):

```
run_dir/videos/
├── episode_000000_right_wrist_camera_0_main.mp4
├── episode_000000_right_wrist_camera_0_depth.mp4
├── episode_000001_right_wrist_camera_0_main.mp4
└── ...
```

With `--separate_successful_episodes`:

```
run_dir/videos/
├── successful/
│   ├── episode_000001_right_wrist_camera_0_main.mp4
│   └── ...
├── failed/
│   └── ...
└── unlabeled/
    └── ...
```

---