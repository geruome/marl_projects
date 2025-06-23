#!/bin/sh

# 检查参数数量
[ "$#" -ne 2 ] && { echo "Usage: $0 <start_pid> <end_pid>"; exit 1; }

# 遍历 PID 范围并尝试杀死
for pid in $(seq "$1" "$2"); do
    kill -9 "$pid" > /dev/null 2>&1 && echo "Killed $pid" || echo "Failed to kill $pid (might not exist)"
done