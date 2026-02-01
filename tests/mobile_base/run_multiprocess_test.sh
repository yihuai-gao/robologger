#!/bin/bash

echo "====================================================="
echo "Starting MobileBaseLogger Multiprocess Test"
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

echo ""
echo "All child loggers started. Starting main logger..."
echo ""
sleep 1

python tests/mobile_base/test_main_logger.py

echo ""
echo "Main logger finished. Cleaning up child processes..."
kill $PID1 $PID2 $PID3 2>/dev/null
wait

echo ""
echo "====================================================="
echo "Test completed!"
echo "====================================================="
