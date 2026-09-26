"""Teach waypoints by moving the SO-101 follower by hand (no leader arm).

Workflow:
  1. Stop any rollout / record script so the port is free.
  2. Run: python scripts/teach_waypoints_by_hand.py --out outputs/waypoints/tape_grasp.json
  3. Torque is disabled -> arm goes limp. SUPPORT THE ARM so it doesn't drop.
  4. Move it by hand to each pose, press ENTER to capture.
  5. Replay with: python scripts/replay_waypoints.py --in outputs/waypoints/tape_grasp.json

Only repeatable setups work (same tape position, same arm base).
"""
import argparse
import json
import time
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

DEFAULT_NAMES = [
    "above_tape_gripper_open",
    "lowered_around_tape",
    "gripper_closed",
    "lifted_with_tape",
    "return_release",
]

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

def read_joints(robot):
    vals = robot.bus.sync_read("Present_Position", num_retry=2)
    return {f"{j}.pos": float(vals[j]) for j in JOINTS}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/tty.usbmodem5C821089701")
    ap.add_argument("--id", default="my_awesome_follower_arm")
    ap.add_argument("--out", default="outputs/waypoints/tape_grasp.json")
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    robot = make_robot(args.port, args.id)
    print(f"Connecting to {args.port} (calibrate=False)...", flush=True)
    robot.connect(calibrate=False)
    try:
        robot.bus.disable_torque()
        print("\nTORQUE DISABLED - arm is limp. SUPPORT IT so it doesn't fall.", flush=True)
        print("Move the arm BY HAND to each pose. Press ENTER to capture it.", flush=True)
        print("Type 'q' + ENTER to finish early, 'r' + ENTER to redo last.\n", flush=True)
        time.sleep(0.3)

        waypoints = []
        for i, name in enumerate(args.names):
            while True:
                ans = input(f"[{i+1}/{len(args.names)}] Move to '{name}' then ENTER (q=finish, r=redo last): ").strip().lower()
                if ans == "q":
                    print("Finishing early.", flush=True)
                    break
                if ans == "r" and waypoints:
                    dropped = waypoints.pop()
                    print(f"  dropped '{dropped['name']}'. Re-capture it.", flush=True)
                    # re-capture same index: stay in loop but adjust target name
                    # simplest: capture replacement for dropped pose now
                    joints = read_joints(robot)
                    waypoints.append({"name": dropped["name"], "joints": joints})
                    print(f"  saved {dropped['name']}: " + ", ".join(f"{k}={v:.1f}" for k, v in joints.items()), flush=True)
                    break
                # default: ENTER captures
                joints = read_joints(robot)
                waypoints.append({"name": name, "joints": joints})
                print(f"  saved {name}: " + ", ".join(f"{k}={v:.1f}" for k, v in joints.items()), flush=True)
                break
            if ans == "q":
                break

        if not waypoints:
            print("No waypoints captured, nothing saved.", flush=True)
            return
        payload = {
            "port": args.port,
            "id": args.id,
            "use_degrees": True,
            "joint_order": [f"{j}.pos" for j in JOINTS],
            "waypoints": waypoints,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nSaved {len(waypoints)} waypoints -> {out}", flush=True)
        for w in waypoints:
            print(f"  - {w['name']}", flush=True)
        print("\nReplay with:", flush=True)
        print(f"  python scripts/replay_waypoints.py --in {out}", flush=True)
    finally:
        try:
            robot.bus.enable_torque()
            print("Torque re-enabled (arm holds). Support it on disconnect.", flush=True)
        except Exception as e:
            print(f"Could not re-enable torque: {e}", flush=True)
        robot.disconnect()

if __name__ == "__main__":
    main()
