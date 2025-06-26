#!/bin/sh

# Check if only one argument is provided (the starting PID)
[ "$#" -ne 1 ] && { echo "Usage: $0 <start_pid>"; exit 1; }

start_pid="$1"
end_pid=$((start_pid + 5000)) # Calculate the end_pid

# echo "Attempting to kill processes from PID $start_pid to PID $end_pid (inclusive)."
# echo "WARNING: Using kill -9 (SIGKILL) which is an ungraceful termination."

# Loop through the PID range and attempt to kill each process
for pid in $(seq "$start_pid" "$end_pid"); do
    kill -9 "$pid" > /dev/null 2>&1 
    # && echo "Killed $pid" || echo "Failed to kill $pid (might not exist)"

sudo rm /dev/shm/model-pool
done

# echo "Kill operation completed."