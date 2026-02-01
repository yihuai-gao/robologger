# MobileBaseLogger Usage Guide

## Overview
The MobileBaseLogger logs mobile base data with flexible configuration for pose and velocity logging.

## Data Format
- **Pose**: 3D vector `(pos_x, pos_y, rot_yaw)`
- **Velocity**: 3D vector `(vel_x, vel_y, vel_yaw)`

## Configuration Parameters

```python
MobileBaseLogger(
    name: str,
    endpoint: str,
    attr: Dict[str, Any],
    log_state_pose: bool = True,        # Log state pose
    log_state_velocity: bool = False,   # Log state velocity
    log_target_pose: bool = True,       # Log target pose
    log_target_velocity: bool = False,  # Log target velocity
)
```

**Constraints:**
- At least one of `log_state_pose` or `log_state_velocity` must be True
- At least one of `log_target_pose` or `log_target_velocity` must be True

## Logging Modes

### 1. Pose Only
Logs only position and orientation, no velocity.

```python
logger = MobileBaseLogger(
    name="mobile_base",
    endpoint="tcp://localhost:5555",
    attr={"robot_type": "mobile_base"},
    log_state_pose=True,
    log_state_velocity=False,
    log_target_pose=True,
    log_target_velocity=False
)

logger.log_state(
    state_timestamp=t,
    state_pose=np.array([x, y, yaw])
)
logger.log_target(
    target_timestamp=t,
    target_pose=np.array([x, y, yaw])
)
```

**Data Structure:**
- state_timestamps, state_pose
- target_timestamps, target_pose

### 2. Pose + Velocity
Logs both pose and velocity for state and target.

```python
logger = MobileBaseLogger(
    name="mobile_base",
    endpoint="tcp://localhost:5555",
    attr={"robot_type": "mobile_base"},
    log_state_pose=True,
    log_state_velocity=True,
    log_target_pose=True,
    log_target_velocity=True
)

logger.log_state(
    state_timestamp=t,
    state_pose=np.array([x, y, yaw]),
    state_velocity=np.array([vx, vy, vyaw])
)
logger.log_target(
    target_timestamp=t,
    target_pose=np.array([x, y, yaw]),
    target_velocity=np.array([vx, vy, vyaw])
)
```

**Data Structure:**
- state_timestamps, state_pose, state_velocity
- target_timestamps, target_pose, target_velocity

### 3. Velocity Only (NO Pose)
Logs only velocity data, no pose information.

```python
logger = MobileBaseLogger(
    name="mobile_base",
    endpoint="tcp://localhost:5555",
    attr={"robot_type": "mobile_base"},
    log_state_pose=False,
    log_state_velocity=True,
    log_target_pose=False,
    log_target_velocity=True
)

logger.log_state(
    state_timestamp=t,
    state_velocity=np.array([vx, vy, vyaw])
)
logger.log_target(
    target_timestamp=t,
    target_velocity=np.array([vx, vy, vyaw])
)
```

**Data Structure:**
- state_timestamps, state_velocity
- target_timestamps, target_velocity

### 4. Mixed Mode
Flexible configuration - e.g., state has pose+velocity, target has velocity only.

```python
logger = MobileBaseLogger(
    name="mobile_base",
    endpoint="tcp://localhost:5555",
    attr={"robot_type": "mobile_base"},
    log_state_pose=True,
    log_state_velocity=True,
    log_target_pose=False,
    log_target_velocity=True
)

logger.log_state(
    state_timestamp=t,
    state_pose=np.array([x, y, yaw]),
    state_velocity=np.array([vx, vy, vyaw])
)
logger.log_target(
    target_timestamp=t,
    target_velocity=np.array([vx, vy, vyaw])
)
```

**Data Structure:**
- state_timestamps, state_pose, state_velocity
- target_timestamps, target_velocity

## Running Tests

### Basic Test (3 modes)
```bash
bash tests/run_mobile_base_multiprocess_test.sh
```

### Comprehensive Test (5 modes)
```bash
bash tests/run_mobile_base_all_modes_test.sh
```

## Test Scripts

1. `test_mobile_base_logger_pose_only.py` - Pose only
2. `test_mobile_base_logger_with_velocity.py` - Pose + velocity (both)
3. `test_mobile_base_logger_state_vel_only.py` - State velocity, target pose
4. `test_mobile_base_logger_velocity_only.py` - Velocity only (NO pose)
5. `test_mobile_base_logger_mixed.py` - Mixed mode example
6. `test_mobile_base_main_logger.py` - Main logger for basic test
7. `test_mobile_base_all_modes_main_logger.py` - Main logger for comprehensive test

## Key Features

- **Flexible Configuration**: Independent control for state and target logging
- **In-Memory Buffering**: Low-latency logging with batch zarr write
- **Multiprocess Support**: Full integration with MainLogger
- **Optional Fields**: Only stores data that's configured to be logged
- **Validation**: Automatic validation of required fields based on configuration
