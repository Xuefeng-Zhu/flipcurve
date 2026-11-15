#!/bin/bash
# Phase 2, run from flipcurve/hard/. Stages are strictly sequential so the cohort cannot drift:
#   ./run_hard.sh screen   # repeat (19-min bounded, resumable) until screen counts stop growing
#   ./run_hard.sh freeze   # write manifest.json once, from the final screens
#   ./run_hard.sh args     # generate arguments for the frozen cohort, then audit them
#   ./run_hard.sh push     # repeat until push counts stop growing; refuses to run without manifest.json
set -e
cd "$(dirname "$0")"
python -c "import adaption, pandas" 2>/dev/null || pip install -q adaption datasets pandas matplotlib numpy statsmodels
M="gemma-4-31b-it qwen3.8-27b qwen3.5-397b glm-5.3-flash glm-5.2 deepseek-v4-flash"
case "$1" in
  screen) timeout 1140 python ../hard.py screen --models $M --workers 16 >> log_screen.txt 2>&1 || true ;;
  freeze) python ../hard.py freeze --models $M ;;
  args)   python ../hard.py args && (cd .. && python argfix.py judge --all && python argfix.py regen) ;;
  push)   timeout 1140 python ../hard.py push --models $M --workers 8 >> log_push.txt 2>&1 || true ;;
  *) echo "usage: $0 screen|freeze|args|push"; exit 1 ;;
esac
wc -l screen_*.jsonl runs_*.jsonl 2>/dev/null || true
