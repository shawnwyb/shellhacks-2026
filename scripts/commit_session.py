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
        self._commands["commitandrecord"] = (
            self._cmd_commitandrecord,
            " [seconds] [reset]",
            "same as /commit but records top+wrist video through coarse and primitives",
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

    def _cmd_commitandrecord(self, cmd):
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
            args=(self, want_reset, seconds, True),
            name="commit-record",
            daemon=True,
        ).start()
        self._print(f"Commit+record started — recording cams, Molmo coarse {seconds:g}s, then primitives...")

    session_cls.__init__ = __init__
    session_cls._cmd_commit = _cmd_commit
    session_cls._cmd_commitandrecord = _cmd_commitandrecord
    return session_cls


def _start_cam_recorder(robot, outdir, fps=10.0):
    import threading

    outdir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    (outdir / "meta.json").write_text(
        __import__("json").dumps({"fps": fps, "started": __import__("time").time()})
    )

    def loop():
        import time as _time
        import cv2 as _cv2
        import numpy as _np
        writers = {}
        try:
            interval = 1.0 / max(1.0, fps)
            while not stop.is_set():
                tick = _time.monotonic()
                try:
                    obs = robot.get_observation()
                except Exception:
                    _time.sleep(interval)
                    continue
                for key, tag in (("cam0", "top"), ("cam1", "wrist")):
                    frame = obs.get(key)
                    if frame is None:
                        continue
                    if key not in writers:
                        h, w = frame.shape[:2]
                        path = str(outdir / f"{key}_{tag}.mp4")
                        writers[key] = _cv2.VideoWriter(
                            path, _cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    writers[key].write(_cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR))
                elapsed = _time.monotonic() - tick
                _time.sleep(max(0.0, interval - elapsed))
        finally:
            for w in writers.values():
                try:
                    w.release()
                except Exception:
                    pass

    thread = threading.Thread(target=loop, name="commit-cam-recorder", daemon=True)
    thread.start()
    return stop, thread


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


def _run_commit(session, want_reset, seconds=15.0, record=False):
    ctrl = session.controller
    ctx = session._ctx
    robot = ctx.hardware.robot_wrapper
    rec_stop = None
    rec_thread = None
    rec_dir = None
    try:
        if record:
            from datetime import datetime as _dt
            rec_dir = Path("outputs/commit_records") / _dt.now().strftime("%Y%m%d-%H%M%S-%f")
            rec_fps = float(os.environ.get("COMMIT_RECORD_FPS", "10"))
            rec_stop, rec_thread = _start_cam_recorder(robot, rec_dir, fps=rec_fps)
            session._print(f"Recording cams to {rec_dir} ...")
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
        if record and rec_dir is not None:
            import time as _time2
            hold_s = float(os.environ.get("COMMIT_RECORD_HOLD_S", "5"))
            session._print(f"Recording hold {hold_s:g}s for the grasp...")
            _time2.sleep(max(0.0, hold_s))
        if want_reset:
            session._print("Commit done — resetting to initial position...")
            ctrl.reset()
        else:
            session._print("Commit done — holding. /commit reset to run + return, /reset to return, /start to resume Molmo.")
    except Exception as exc:
        session._print(f"Commit failed: {exc}")
    finally:
        if record and rec_stop is not None:
            rec_stop.set()
            if rec_thread is not None:
                rec_thread.join(timeout=5)
            session._print(f"Recording saved: {rec_dir}")
