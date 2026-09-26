"""FSM commit primitives: down, clamp, up, hold (test).

Loads the dive segment from a taught grasp file and executes it blind:
  lowered_around -> gripper_closed (long hold) -> lifted, then holds position
  with torque on so you can inspect the grasp.

  python scripts/commit_primitives.py --dry-run
  python scripts/commit_primitives.py
  python scripts/commit_primitives.py --grasp outputs/waypoints/tape_grasp.json --hold-seconds 30
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
        max_relative_target=None,
    )
    return SO101Follower(cfg)


def interp(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def move_linear(robot, start, goal, step_deg=1.5, fps=30.0):
    dist = max(abs(goal[k] - start[k]) for k in start)
    steps = max(1, int(dist / step_deg))
    dt = 1.0 / fps
    for i in range(1, steps + 1):
        robot.send_action(interp(start, goal, i / steps))
        time.sleep(dt)
    return goal


def pick(waypoints, *needles, fallback=0):
    for i, w in enumerate(waypoints):
        if all(n in w["name"].lower() for n in needles):
            return i
    return fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grasp", default="outputs/waypoints/tape_grasp.json")
    ap.add_argument("--port", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--step-deg", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dive-step-deg", type=float, default=0.4)
    ap.add_argument("--dive-fps", type=float, default=15.0)
    ap.add_argument("--grip-pause", type=float, default=2.0)
    ap.add_argument("--closed-gripper", type=float, default=0.0)
    ap.add_argument("--hold-seconds", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = json.loads(Path(args.grasp).read_text())
    wps = payload["waypoints"]
    port = args.port or payload.get("port", "/dev/tty.usbmodem5C821089701")
    robot_id = args.id or payload.get("id", "my_awesome_follower_arm")
    lower_i = pick(wps, "lower", fallback=1)
    close_i = pick(wps, "clos", fallback=2)
    lift_i = pick(wps, "lift", fallback=3)
    seq = wps[lower_i:lift_i + 1] if lift_i >= lower_i else wps[1:4]
    for w in seq:
        name = w["name"].lower()
        if any(s in name for s in ("clos", "grip", "lift")):
            w["joints"]["gripper.pos"] = float(args.closed_gripper)

    print(f"Commit primitives from {args.grasp}: {[w['name'] for w in seq]}", flush=True)
    print(f"Hold {args.hold_seconds}s at the end with torque on.", flush=True)
    if args.dry_run:
        return

    robot = make_robot(port, robot_id)
    robot.connect(calibrate=False)
    try:
        robot.bus.enable_torque()
        cur = {f"{j}.pos": float(v) for j, v in robot.bus.sync_read("Present_Position", num_retry=2).items() if j in JOINTS}
        cur = {f"{j}.pos": cur[f"{j}.pos"] for j in JOINTS}
        states = ["APPROACH_DONE", "DIVE", "CLAMP", "LIFT", "HOLD"]
        print(f"FSM: {states[0]} -> {states[1]}", flush=True)
        for w in seq:
            goal = {k: float(v) for k, v in w["joints"].items()}
            name = w["name"].lower()
            is_dive = "lower" in name or "dive" in name
            sd = args.dive_step_deg if is_dive else args.step_deg
            fp = args.dive_fps if is_dive else args.fps
            if "clos" in name or "grip" in name:
                print("FSM: DIVE -> CLAMP", flush=True)
            elif "lift" in name:
                print("FSM: CLAMP -> LIFT", flush=True)
            print(f"-> {w['name']}" + (" (slow)" if (is_dive or "lift" in name) else ""), flush=True)
            if "lift" in name:
                sd, fp = 0.6, 20.0
            cur = move_linear(robot, cur, goal, step_deg=sd, fps=fp)
            if any(s in name for s in ("grip", "clos")):
                time.sleep(1.0)
                robot.send_action(goal)
                time.sleep(3.0)
                cur = move_linear(robot, cur, goal, step_deg=0.4, fps=15.0)
                time.sleep(1.0)
            else:
                time.sleep(0.8)
        print(f"FSM: LIFT -> HOLD ({args.hold_seconds}s, Ctrl-C to release early)", flush=True)
        robot.send_action(cur)
        time.sleep(args.hold_seconds)
        print("FSM: HOLD done.", flush=True)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
