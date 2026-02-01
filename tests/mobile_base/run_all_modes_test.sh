#!/bin/bash

echo "====================================================="
echo "Starting MobileBaseLogger All Modes Test"
echo "====================================================="

cleanup() {
    echo ""
    echo "Cleaning up processes..."
    pkill -P $$
    exit
}

trap cleanup SIGINT SIGTERM

echo ""
echo "Starting child logger processes..."
echo ""

python tests/mobile_base/test_pose_only.py &
PID1=$!
echo "[LAUNCHER] Started pose_only logger (PID: $PID1)"
sleep 0.5

python tests/mobile_base/test_with_velocity.py &
PID2=$!
echo "[LAUNCHER] Started with_velocity logger (PID: $PID2)"
sleep 0.5

python tests/mobile_base/test_state_vel_only.py &
PID3=$!
echo "[LAUNCHER] Started state_vel_only logger (PID: $PID3)"
sleep 0.5

python tests/mobile_base/test_velocity_only.py &
PID4=$!
echo "[LAUNCHER] Started velocity_only logger (PID: $PID4)"
sleep 0.5

python tests/mobile_base/test_mixed.py &
PID5=$!
echo "[LAUNCHER] Started mixed logger (PID: $PID5)"
sleep 0.5

echo ""
echo "All child loggers started. Starting main logger..."
echo ""
sleep 1

python tests/mobile_base/test_all_modes_main_logger.py

echo ""
echo "Main logger finished. Cleaning up child processes..."
kill $PID1 $PID2 $PID3 $PID4 $PID5 2>/dev/null
wait

echo ""
echo "====================================================="
echo "Test completed!"
echo "====================================================="
