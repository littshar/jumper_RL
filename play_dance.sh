#!/usr/bin/env bash
# Replay the latest dance_sway checkpoint. Usage: play_dance.sh [model.pt]
cd "$(dirname "$0")"
CKPT="${1:-logs/jumper/jumper.dance_sway/2026-10-17_17-50-02/model_17649.pt}"
exec .venv/bin/python scripts/play.py --task jumper.dance_sway --checkpoint "$CKPT"
