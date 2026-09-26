#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

inference_args=(--inference.type=sync)
mode_label="synchronous inference, 30 actions per observation"
if [[ $# -eq 1 && "$1" == "--prefetch" ]]; then
  inference_args=(--inference.type=rtc --inference.rtc.enabled=false --inference.queue_threshold=30)
  mode_label="prefetch enabled, queued 30-action sequences"
elif [[ $# -eq 1 && "$1" == "--replan" ]]; then
  # Execute a short prefix, then observe and replan while the arm holds position.
  inference_args=(--inference.type=sync --policy.n_action_steps=8)
  mode_label="replan: prefetch OFF, 8 actions per observation"
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/run_molmo.sh [--prefetch|--replan]" >&2
  exit 2
fi

echo "Rollout mode: $mode_label"

exec python scripts/molmo_diagnostic_rollout.py \
  --strategy.type=base \
  "${inference_args[@]}" \
  --policy.path=models/molmoact2-mps \
  --policy.inference_action_mode=continuous \
  --policy.device=mps \
  --policy.dtype=bfloat16 \
  --policy.enable_inference_cuda_graph=false \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5C821089701 \
  --robot.id=my_awesome_follower_arm \
  --robot.use_degrees=true \
  --robot.max_relative_target=10 \
  --robot.cameras='{"cam0":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"cam1":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
  --fps=30 \
  --task="Pick up the red tape roll." \
  --duration=30 \
  --interactive=true \
  --return_to_initial_position=false \
  --play_sounds=false
