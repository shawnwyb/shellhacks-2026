"""Replay hand-taught joint waypoints on the SO-101 follower.

  python scripts/replay_waypoints.py --in outputs/waypoints/tape_grasp.json

Slowly interpolates between poses. Pause at each waypoint; gripper
poses (name contains open/close/grip) hold longer so the grasp settles.
Only works for a repeatable setup - same object spot, same arm base.
"""
import argparse
import json
import time
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

def make_robot(port, robot_id):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
        cameras={},
        max_relative_target=None,  # we interpolate manually; don't let clipping fight us
    )
    return SO101Follower(cfg)

def interp(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t for k in a}

def move_linear(robot, start, goal, step_deg=1.5, fps=30):
    dist = max(abs(goal[k] - start[k]) for k in start)
    steps = max(1, int(dist / step_deg))
    dt = 1.0 / fps
    for i in range(1, steps + 1):
        robot.send_action(interp(start, goal, i / steps))
        time.sleep(dt)
    return goal

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="waypoints JSON from teach script")
    ap.add_argument("--port", default=None, help="override port (default: from file or follower port)")
    ap.add_argument("--id", default=None, help="override robot id")
    ap.add_argument("--step-deg", type=float, default=1.5, help="max joint degrees per interpolation step")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pause", type=float, default=0.8, help="pause at each waypoint (s)")
    ap.add_argument("--grip-pause", type=float, default=1.5, help="extra hold for gripper poses (s)")
    ap.add_argument("--dry-run", action="store_true", help="print waypoints without moving")
    ap.add_argument("--rest", default="outputs/waypoints/rest.json", help="rest pose file for auto-reset after replay")
    ap.add_argument("--no-auto-reset", action="store_true", help="skip auto-reset to rest after replay")
    args = ap.parse_args()

    payload = json.loads(Path(args.inp).read_text())
    waypoints = payload["waypoints"]
    port = args.port or payload.get("port", "/dev/tty.usbmodem5C821089701")
    robot_id = args.id or payload.get("id", "my_awesome_follower_arm")

    print(f"Loaded {len(waypoints)} waypoints from {args.inp}:", flush=True)
    for w in waypoints:
        print(f"  - {w['name']}: " + ", ".join(f"{k}={w['joints'][k]:.1f}" for k in sorted(w["joints"])), flush=True)
    if args.dry_run:
        return

    robot = make_robot(port, robot_id)
    print(f"\nConnecting to {port}...", flush=True)
    robot.connect(calibrate=False)
    try:
        robot.bus.enable_torque()
        cur = {f"{j}.pos": float(v) for j, v in robot.bus.sync_read("Present_Position", num_retry=2).items() if j in JOINTS}
        # normalize key order
        cur = {f"{j}.pos": cur[f"{j}.pos"] for j in JOINTS}
        print("Moving to first waypoint slowly...", flush=True)
        first = {k: float(v) for k, v in waypoints[0]["joints"].items()}
        cur = move_linear(robot, cur, first, step_deg=args.step_deg, fps=args.fps)
        time.sleep(args.pause)

        for w in waypoints[1:]:
            goal = {k: float(v) for k, v in w["joints"].items()}
            print(f"-> {w['name']}", flush=True)
            cur = move_linear(robot, cur, goal, step_deg=args.step_deg, fps=args.fps)
            name = w["name"].lower()
            hold = args.grip_pause if any(s in name for s in ("grip", "close", "open", "grasp")) else args.pause
            time.sleep(hold)
        print("Replay done.", flush=True)
        if not args.no_auto_reset:
            rest_path = Path(args.rest)
            if rest_path.is_file():
                rest = json.loads(rest_path.read_text())
                goal = {k: float(v) for k, v in rest["joints"].items()}
                print(f"Auto-reset to rest pose ({rest_path})...", flush=True)
                move_linear(robot, cur, goal, step_deg=args.step_deg, fps=args.fps)
                print("At rest pose.", flush=True)
            else:
                print(f"No rest file at {rest_path}, skipping auto-reset. Save one with:", flush=True)
                print("  python scripts/save_rest_position.py", flush=True)
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
