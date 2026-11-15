#!/bin/bash
# Phase 2, one ~19-min bounded pass: screen and push run side by side; rerun until counts plateau.
cd "$(dirname "$0")"
python -c "import adaption, pandas" 2>/dev/null || pip install -q adaption datasets pandas matplotlib
M="gemma-4-31b-it qwen3.8-27b qwen3.5-397b glm-5.3-flash glm-5.2 deepseek-v4-flash"
timeout 1140 python ../hard.py screen --models $M --workers 16 >> log_screen.txt 2>&1 &
timeout 1140 python ../hard.py push --models $M --workers 8 >> log_push.txt 2>&1 &
wait
wc -l screen_*.jsonl runs_*.jsonl 2>/dev/null
