#!/bin/bash
# one 19-min round across all models; resumable
cd "$(dirname "$0")"
for m in glm-5.2 qwen3.5-397b gemma-4-31b-it qwen3.8-27b glm-5.3-flash; do
  FLIP_RUNS=runs_$m.jsonl timeout 1140 python flipcurve.py run --models $m --workers 5 > log_$m.txt 2>&1 &
done
wait
wc -l runs_*.jsonl
grep -c ERR log_*.txt
