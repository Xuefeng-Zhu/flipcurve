#!/bin/bash
# Phase 2 pipeline, one ~19-min bounded pass; rerun until push counts stop growing.
cd "$(dirname "$0")"
python -c "import adaption, pandas" 2>/dev/null || pip install -q adaption datasets pandas matplotlib
M="gemma-4-31b-it qwen3.8-27b qwen3.5-397b glm-5.3-flash glm-5.2 deepseek-v4-flash"
timeout 1140 bash -c "
python ../hard.py screen --models $M --workers 6 >> log_screen.txt 2>&1
python ../hard.py push --models $M --workers 6 >> log_push.txt 2>&1
"
wc -l screen_*.jsonl runs_*.jsonl arguments.jsonl 2>/dev/null
