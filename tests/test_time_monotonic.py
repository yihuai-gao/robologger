import time

# start_time_monotonic = time.monotonic()
# start_time_time = time.time()
base_time_difference = time.time() - time.monotonic()
while True:
    time_difference = time.time() - time.monotonic()
    print(f"{time_difference - base_time_difference:+.6f}")
