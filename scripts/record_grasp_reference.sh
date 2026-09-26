#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

leader_port="${LEADER_PORT:-/dev/tty.usbmodem5C821079131}"
follower_port=/dev/tty.usbmodem5C821089701
for port in "$leader_port" "$follower_port"; do
  if [[ ! -e "$port" ]]; then
    echo "Missing arm port: $port. Connect both arms; set LEADER_PORT if its port changed." >&2
    exit 1
  fi
done
recording_root="outputs/grasp_reference/$(date +%Y%m%d-%H%M%S)-$$"
echo "Record one successful approach, grasp, and lift. Keep the cameras and object placement unchanged."
echo "Stop the rollout and camera preview programs first. The follower will follow the leader when recording starts."
echo "Saving 30 seconds locally to $recording_root (no upload)."

exec python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port="$follower_port" \
  --robot.id=my_awesome_follower_arm \
  --robot.use_degrees=true \
  --robot.max_relative_target=10 \
  --robot.cameras='{"cam0":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"cam1":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
  --teleop.type=so101_leader \
  --teleop.port="$leader_port" \
  --teleop.id=my_awesome_leader_arm \
  --teleop.use_degrees=true \
  --dataset.repo_id=local/red_tape_reference \
  --dataset.root="$recording_root" \
  --dataset.single_task="Pick up the red tape roll." \
  --dataset.fps=20 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=0 \
  --dataset.num_episodes=1 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --play_sounds=false
