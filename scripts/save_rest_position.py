"""Save the current SO-101 joint pose as the rest state.

  python scripts/save_rest_position.py --out outputs/waypoints/rest.json

Put the arm where you want 'home' to be, then run this. Torque stays
on; it only reads. Used by reset_to_rest.py and auto-reset in replay.
"""
import argparse
import json
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/tty.usbmodem5C821089701")
    ap.add_argument("--id", default="my_awesome_follower_arm")
    ap.add_argument("--out", default="outputs/waypoints/rest.json")
    args = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras={}, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"Connecting to {args.port} (read-only, torque stays on)...", flush=True)
    robot.connect(calibrate=False)
    try:
        vals = robot.bus.sync_read("Present_Position", num_retry=2)
        joints = {f"{j}.pos": float(vals[j]) for j in JOINTS}
    finally:
        robot.disconnect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "port": args.port, "id": args.id, "use_degrees": True,
        "name": "rest", "joints": joints,
        "joint_order": [f"{j}.pos" for j in JOINTS],
    }, indent=2) + "\n")
    print(f"Saved rest pose -> {out}", flush=True)
    print(", ".join(f"{k}={v:.1f}" for k, v in joints.items()), flush=True)

if __name__ == "__main__":
    main()
