# RoboLogger Examples

Examples showing how to integrate RoboLogger into a robot system.

## Architecture

RoboLogger uses a distributed design:
- **Control loops** run independently at their own frequencies
- **Main process** coordinates recording via RobotMQ commands
- Data is only logged when recording is active

```
Main Process (where you setup keyboard control: s/e/y/n/q...)
    │
    ├─── Camera Loop (e.g.30 Hz)
    ├─── Robot Loop (e.g.125 Hz)
    └─── Gripper Loop (e.g.30 Hz)
```

## Files

| File | Data Entries (.zarr) - Example Config | Logger Type |
|------|---------|-------------|
| `main_process.py` | `metadata.zarr/` (.zattrs: project_name, task_name, run_name, morphology, is_successful) | MainLogger |
| `camera_loop.py` | `right_wrist_camera_0.zarr/` (3× .mp4: main, ultrawide, depth + 3× timestamps)* | VideoLogger |
| `robot_control_loop.py` | `right_arm.zarr/` (4× pose arrays: state/target pos_xyz, quat_wxyz + 2× timestamps) | CtrlLogger (EEF control) |
| `joint_arm_control_loop.py` | `right_arm.zarr/` (8× arrays: state/target pos_xyz, quat_wxyz, joint_pos + 2× timestamps) | CtrlLogger (Joint + EEF) |
| `gripper_control_loop.py` | `right_end_effector.zarr/` (2× joint arrays: state/target joint_pos + 2× timestamps) | CtrlLogger (Joint control) |

*Number of cameras, videos, and arrays depends on your configuration

## Quick Start

**Terminal 1-3+**: Start control loops (choose robot example based on your setup)
```bash
python examples/camera_loop.py

# Choose ONE of these robot control examples:
python examples/robot_control_loop.py        # Cartesian/EEF-controlled arm
python examples/joint_arm_control_loop.py    # Joint-controlled arm (most common)

python examples/gripper_control_loop.py
```

**Terminal 4**: Start main process
```bash
python examples/main_process.py
```

**Keyboard controls**:
- `s` - Start recording
- `e` - End recording
- `d` - Delete last episode
- `y` - Mark last episode as successful
- `n` - Mark last episode as failed
- `q` - Quit

## Data Output

```
data/demo_project/demo_task/run_001/episode_000000/
├── metadata.zarr/              # Episode info
├── right_wrist_camera_0.zarr/  # Videos (MP4) + timestamps
├── right_arm.zarr/             # Pose data (state + target) + timestamps
└── right_end_effector.zarr/    # Joint data (state + target) + timestamps
```

## Adapting to Your System

### 1. Replace simulated data with your robot API
```python
# camera_loop.py - lines 70-81
image = camera.capture()  # Replace simulated rainbow frames

# robot_control_loop.py - lines 50-53
state_pos_xyz, state_quat_wxyz = robot.get_eef_pose()

# gripper_control_loop.py - lines 48-52
state_joint_pos = gripper.get_width()

# joint_arm_control_loop.py - lines 52-57
state_joint_pos = robot.get_joint_positions()
state_pos_xyz, state_quat_wxyz = robot.forward_kinematics(state_joint_pos)
```

### 2. Match endpoints across files
`main_process.py` endpoints must match control loop endpoints (default: ports 55555-55557).

### 3. Use valid logger names
Names must match enums in `robologger/utils/classes.py`:
- **Cameras**: `right_wrist_camera_0`, `head_camera_0` (continuous and zero-indexed)
- **Robots**: `right_arm`, `left_arm`, `right_end_effector`, `left_end_effector`

## Success Labeling

Configure `success_config` in `MainLogger`:
- `"none"` - Does not set is_successful (no prompt, no value assigned)
- `"input_true"` - Prompt user with [Y/n], defaults to successful
- `"input_false"` - Prompt user with [y/N], defaults to failed
- `"hardcode_true"` - Always mark episodes as successful (no prompt)
- `"hardcode_false"` - Always mark episodes as failed (no prompt)

## Key Concepts

- Control loops run continuously, only log when recording is active
- Each loop runs at its own frequency independently
- All timestamps use `time.monotonic()` for consistency
- State = actual (measured), Target = commanded

## CtrlLogger Configuration

The unified `CtrlLogger` supports flexible control configurations:

### Configuration Options
- `log_eef_pose`: Log end-effector pose (position + quaternion)
- `log_joint_positions`: Log joint positions
- `target_type`: Control target type (`"eef_pose"` or `"joint_positions"`)
- `joint_units`: Joint units (`"radians"` or `"meters"`, or `None` if not logging joints)

### Common Use Cases

**1. EEF-controlled arm (Cartesian control)**
```python
CtrlLogger(
    name="right_arm",
    log_eef_pose=True,
    log_joint_positions=False,
    target_type="eef_pose",
    joint_units=None
)
# Log: state_pos_xyz, state_quat_wxyz, target_pos_xyz, target_quat_wxyz
```

**2. Joint-controlled arm that can infer EEF pose (most common)**
```python
CtrlLogger(
    name="left_arm",
    attr={"num_joints": 7},
    log_eef_pose=True,          # Log inferred EEF pose via forward kinematics
    log_joint_positions=True,   # Log joint positions
    target_type="joint_positions",
    joint_units="radians"
)
# Log: all EEF pose fields + state_joint_pos, target_joint_pos
```

**3. Joint-controlled gripper (no EEF pose)**
```python
CtrlLogger(
    name="right_end_effector",
    attr={"num_joints": 1},
    log_eef_pose=False,
    log_joint_positions=True,
    target_type="joint_positions",
    joint_units="meters"        # Gripper width in meters
)
# Log: state_joint_pos, target_joint_pos only
```

**4. EEF-controlled arm with joint encoders**
```python
CtrlLogger(
    name="right_arm",
    attr={"num_joints": 7},
    log_eef_pose=True,          # Primary control mode
    log_joint_positions=True,   # Also log observed joint positions
    target_type="eef_pose",
    joint_units="radians"
)
# Log: all fields (EEF pose + joint positions)
```