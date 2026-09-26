"""Move the SO-101 back to the saved rest pose.

  python scripts/reset_to_rest.py --rest outputs/waypoints/rest.json

Slow linear interpolation, same motion style as replay_waypoints.py.
"""
import argparse
import json
import time
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

def move_linear(robot, start, goal, step_deg=1.5, fps=30):
    dist = max(abs(goal[k] - start[k]) for k in start)
    steps = max(1, int(dist / step_deg))
    dt = 1.0 / fps
    for i in range(1, steps + 1):
        t = i / steps
        robot.send_action({k: start[k] + (goal[k] - start[k]) * t for k in start})
        time.sleep(dt)
    return goal

def load_rest(path):
    data = json.loads(Path(path).read_text())
    if "joints" in data:
        return data
    if "waypoints" in data and data["waypoints"]:
        w = data["waypoints"][0]
        return {"port": data.get("port"), "id": data.get("id"), "joints": w["joints"]}
    raise ValueError(f"No joints found in {path}")

def goto_rest(rest_path, port=None, robot_id=None, step_deg=1.5, fps=30):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    rest = load_rest(rest_path)
    port = port or rest.get("port", "/dev/tty.usbmodem5C821089701")
    robot_id = robot_id or rest.get("id", "my_awesome_follower_arm")
    goal = {k: float(v) for k, v in rest["joints"].items()}
    cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True, cameras={}, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"Resetting to rest pose from {rest_path}...", flush=True)
    robot.connect(calibrate=False)
    try:
        robot.bus.enable_torque()
        vals = robot.bus.sync_read("Present_Position", num_retry=2)
        cur = {f"{j}.pos": float(vals[j]) for j in JOINTS}
        move_linear(robot, cur, goal, step_deg=step_deg, fps=fps)
        print("At rest pose.", flush=True)
    finally:
        robot.disconnect()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest", default="outputs/waypoints/rest.json")
    ap.add_argument("--port", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--step-deg", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    goto_rest(args.rest, port=args.port, robot_id=args.id, step_deg=args.step_deg, fps=args.fps)

if __name__ == "__main__":
    main()
