#!/bin/bash

# Number of times to repeat the each experiment
rounds=10

# Array containing the sizes
init_size=216
final_size=408

# Array containing the times
times=(0.02 0.1 1.0)

# Array containing the versions
versions=("original" "original-parallel-write-single-file" "original-parallel-write-part-files")

nvidia-smi --format=csv --loop-ms=100 --query-gpu=timestamp,name,uuid,pstate,memory.total,memory.used,memory.free,temperature.gpu,utilization.memory,utilization.gpu,power.management,power.draw >> gpu_power.csv &

# Nested for-loops to iterate over sizes, times, and versions
for round in $(seq 1 $((rounds))); do
  for time in "${times[@]}"; do
    for size in $(seq $((init_size)) 32 $((final_size))); do
      for version in "${versions[@]}"; do
        echo "Processing round: $round size: $size, time: $time, version: $version"
        # echo "./${version}/ModelagemFletcher.exe TTI $size $size $size 16 12.5 12.5 12.5 0.001 $time"
        output=$(perf stat -e power/energy-pkg/ -x, -o /tmp/cpu_power.csv ./${version}/ModelagemFletcher.exe TTI $size $size $size 16 12.5 12.5 12.5 0.001 $time | grep "$version")
        joules=$(tail -n +3 /tmp/cpu_power.csv | cut -d, -f1)
        output="$output,$joules"
        echo "$output"
        echo "$output" >> time.csv
      done
    done
  done
done

killall nvidia-smi
