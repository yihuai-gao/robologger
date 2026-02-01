# Mobile Base Logger Tests

This directory contains all test files for the MobileBaseLogger implementation.

## Directory Structure

```
tests/mobile_base/
├── README.md                          # This file
├── MOBILE_BASE_LOGGER_USAGE.md        # Detailed usage guide
├── run_multiprocess_test.sh           # Basic test (3 modes)
├── run_all_modes_test.sh              # Comprehensive test (5 modes)
├── test_pose_only.py                  # Pose only mode test
├── test_with_velocity.py              # Pose + velocity mode test
├── test_state_vel_only.py             # State velocity only test
├── test_velocity_only.py              # Velocity only (no pose) test
├── test_mixed.py                      # Mixed mode test
├── test_main_logger.py                # Main logger for basic test
└── test_all_modes_main_logger.py      # Main logger for comprehensive test
```

## Running Tests

### Quick Test (3 Modes)
Tests basic functionality with pose-only, pose+velocity, and state velocity modes:
```bash
bash tests/mobile_base/run_multiprocess_test.sh
```

### Comprehensive Test (5 Modes)
Tests all logging modes including velocity-only and mixed configurations:
```bash
bash tests/mobile_base/run_all_modes_test.sh
```

## Test Modes

1. **Pose Only** - Logs only position and orientation
2. **With Velocity** - Logs both pose and velocity for state and target
3. **State Velocity Only** - State has velocity, target has pose only
4. **Velocity Only** - NO pose data, velocity only (demonstrates flexibility)
5. **Mixed Mode** - State has pose+velocity, target has velocity only

## Key Features Tested

- ✅ In-memory buffering with batch zarr write
- ✅ Multiprocess coordination via MainLogger
- ✅ Flexible configuration for state and target logging
- ✅ Independent control of pose and velocity logging
- ✅ Velocity-only logging (without pose data)
- ✅ Data validation and error handling
- ✅ Proper zarr data structure and metadata

## Usage

See `MOBILE_BASE_LOGGER_USAGE.md` for detailed usage examples and configuration options.
