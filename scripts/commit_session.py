"""Interactive /commit support for the Molmo rollout.

Patched into InteractiveSession by scripts/molmo_diagnostic_rollout.py:
  /commit           run Molmo coarse 15s, then FSM primitives down/clamp/up, hold
  /commit 20        same with 20s Molmo coarse
  /commit reset     same, then /reset back to initial position
  /commit 20 reset  20s Molmo coarse, primitives, then reset

Primitives come from outputs/waypoints/tape_grasp.json so the checkpoint
matches your hand-taught dive. Override with COMMIT_GRASP env var.
"""
import json
import os
import time
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def load_commit_seq(grasp_path=None, closed_gripper=None):
    import os as _os
    if closed_gripper is None:
        try:
            closed_gripper = float(_os.environ.get("COMMIT_CLOSED", "0.0"))
        except ValueError:
            closed_gripper = 0.0
    path = Path(grasp_path or os.environ.get("COMMIT_GRASP", "outputs/waypoints/tape_grasp.json"))
    wps = json.loads(path.read_text())["waypoints"]

    def pick(*needles, fallback=0):
        for i, w in enumerate(wps):
            if all(n in w["name"].lower() for n in needles):
                return i
        return fallback

    lower_i = pick("lower", fallback=1)
    close_i = pick("clos", fallback=2)
    lift_i = pick("lift", fallback=3)
    seq = wps[lower_i:lift_i + 1] if lift_i >= lower_i else wps[1:4]
    out = []
    for w in seq:
        d = {k: float(v) for k, v in w["joints"].items()}
        name = w["name"].lower()
        if any(s in name for s in ("clos", "grip", "lift")):
            d["gripper.pos"] = float(closed_gripper)
        out.append(d | {"_name": w["name"]})
    return out


def patch_session(session_cls):
    orig_init = session_cls.__init__

    def __init__(self, strategy, ctx, input_stream=None):
        orig_init(self, strategy, ctx, input_stream=input_stream)
        self._ctx = ctx
        self._commands["commit"] = (
            self._cmd_commit,
            " [seconds] [reset]",
            "run Molmo coarse (default 15s) then taught down/clamp/up primitives and hold",
        )

    def _cmd_commit(self, cmd):
        import threading

        parts = cmd.args.strip().lower().split()
        want_reset = "reset" in parts
        seconds = float(os.environ.get("COMMIT_SECONDS", "15"))
        for part in parts:
            try:
                seconds = float(part)
                break
            except ValueError:
                continue
        if getattr(self.controller, "stopped", False):
            self._print("Can't commit — the session has stopped.")
            return
        threading.Thread(
            target=_run_commit,
            args=(self, want_reset, seconds),
            name="commit-primitives",
            daemon=True,
        ).start()
        self._print(f"Commit started — Molmo coarse {seconds:g}s, then primitives taking over...")

    session_cls.__init__ = __init__
    session_cls._cmd_commit = _cmd_commit
    return session_cls


def _joints_from_obs(obs):
    cur = {}
    for j in JOINTS:
        key = f"{j}.pos"
        if key in obs:
            cur[key] = float(obs[key])
    return cur


def _move(robot, start, goal, step_deg=1.5, fps=30.0):
    dist = max(abs(goal[k] - start[k]) for k in start)
    steps = max(1, int(dist / step_deg))
    dt = 1.0 / fps
    keys = [k for k in goal if not k.startswith("_")]
    for i in range(1, steps + 1):
        robot.send_action({k: start[k] + (goal[k] - start[k]) * i / steps for k in keys})
        time.sleep(dt)
    return {k: goal[k] for k in keys}


def _run_commit(session, want_reset, seconds=15.0):
    ctrl = session.controller
    ctx = session._ctx
    robot = ctx.hardware.robot_wrapper
    try:
        if not ctrl._running.is_set():
            if not ctrl.start():
                session._print("Commit: policy already running elsewhere, using live motion...")
            else:
                session._print(f"Commit: Molmo coarse running {seconds:g}s...")
        else:
            session._print(f"Commit: using live Molmo motion for {seconds:g}s...")
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if getattr(ctrl, "stopped", False):
                session._print("Commit aborted — session stopped.")
                return
            time.sleep(0.1)
        session._print("Commit: stopping policy, primitives taking over...")
        ctrl._start_requested.clear()
        ctrl._segment_stop.set()
        ctrl._wake.set()
        for _ in range(100):
            if not ctrl._running.is_set():
                break
            time.sleep(0.05)
        time.sleep(0.3)
        seq = load_commit_seq()
        import os as _os
        try:
            dive_step = float(_os.environ.get("COMMIT_DIVE_STEP_DEG", "0.4"))
        except ValueError:
            dive_step = 0.4
        try:
            dive_fps = float(_os.environ.get("COMMIT_DIVE_FPS", "15.0"))
        except ValueError:
            dive_fps = 15.0
        session._print(f"Commit dive speed: {dive_step:g} deg/step at {dive_fps:g} Hz (slow).")
        obs = robot.get_observation()
        cur = _joints_from_obs(obs)
        session._print(f"FSM: APPROACH_DONE -> DIVE ({seq[0].get('_name')})")
        for goal in seq:
            name = str(goal.get("_name", "")).lower()
            is_dive = "lower" in name or "dive" in name
            if "clos" in name or "grip" in name:
                session._print("FSM: DIVE -> CLAMP")
            elif "lift" in name:
                session._print("FSM: CLAMP -> LIFT")
            is_lift = "lift" in name
            cur = _move(robot, cur, goal, step_deg=dive_step if (is_dive or is_lift) else 1.5, fps=dive_fps if (is_dive or is_lift) else 30.0)
            if any(s in name for s in ("grip", "clos")):
                session._print("FSM: CLAMP settling (3s), re-squeezing...")
                time.sleep(1.0)
                robot.send_action({k: goal[k] for k in goal if not k.startswith("_")})
                time.sleep(3.0)
                cur = _move(robot, cur, goal, step_deg=0.4, fps=15.0)
                time.sleep(1.0)
            else:
                time.sleep(0.8)
        session._print("FSM: LIFT -> HOLD (holding position)")
        robot.send_action(cur)
        if want_reset:
            session._print("Commit done — resetting to initial position...")
            ctrl.reset()
        else:
            session._print("Commit done — holding. /commit reset to run + return, /reset to return, /start to resume Molmo.")
    except Exception as exc:
        session._print(f"Commit failed: {exc}")
