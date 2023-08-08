#!/bin/bash

# Array containing the sizes
sizes=(216 600 1000)

# Array containing the times
times=(0.02 0.1)

# Array containing the versions
versions=("original" "original-parallel-write-single-file" "original-parallel-write-part-files")

nvidia-smi --format=csv --loop-ms=100 --query-gpu=timestamp,name,uuid,pstate,memory.total,memory.used,memory.free,temperature.gpu,utilization.memory,utilization.gpu,power.management,power.draw >> power.csv &

# Nested for-loops to iterate over sizes, times, and versions
for time in "${times[@]}"; do
  for size in "${sizes[@]}"; do
    for version in "${versions[@]}"; do
      echo "Processing size: $size, time: $time, version: $version"
      ./${version}/ModelagemFletcher.exe TTI $size $size $size 16 12.5 12.5 12.5 0.001 $time | grep "$version" >> time.csv
    done
  done
done

killall nvidia-smi
