"""Run the normal rollout while recording joint targets and existing observations.

No extra serial reads, camera reads, or motor commands are performed.
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


def install_trace(robot_class, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / "actions.jsonl").open("w", buffering=1)
    original_observe = robot_class.get_observation
    original_send = robot_class.send_action
    latest = {}
    previous_command = None
    frame_saved = False
    frame_directory = directory / "frames"
    frame_directory.mkdir()
    frame_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diagnostic_frames")
    pending_frame = None
    last_frame_time = float("-inf")

    def save_frames(frames, captured_at):
        from PIL import Image

        try:
            for name, frame in frames.items():
                path = frame_directory / f"{captured_at}_{name}.jpg"
                temporary = path.with_suffix(".tmp")
                Image.fromarray(frame).save(temporary, format="JPEG", quality=90)
                temporary.replace(path)
        except Exception:
            logging.exception("Could not save diagnostic camera snapshot")

    def observe(robot):
        nonlocal pending_frame, last_frame_time
        observation = original_observe(robot)
        observed_at = time.monotonic()
        latest[id(robot)] = (observed_at, observation)
        # Reuse existing RGB observations. Opening another capture on macOS can
        # renegotiate the shared camera stream and break the rollout's dimensions.
        # Keep at most one write pending so disk delays cannot build a backlog.
        if observed_at - last_frame_time >= 0.5 and (pending_frame is None or pending_frame.done()):
            frames = {name: observation[name].copy() for name in ("cam0", "cam1") if name in observation}
            if frames:
                pending_frame = frame_writer.submit(save_frames, frames, time.time_ns())
                last_frame_time = observed_at
        return observation

    def send(robot, action):
        nonlocal previous_command, frame_saved
        requested = {k: float(v) for k, v in action.items() if k.endswith(".pos")}
        started = time.monotonic()
        sent = original_send(robot, action)
        observed_at, observation = latest.get(id(robot), (None, {}))
        measured = {k: float(v) for k, v in observation.items() if k.endswith(".pos")}
        transmitted = {k: float(v) for k, v in sent.items() if k.endswith(".pos")}
        record = {
            "schema_version": 2,
            "time": time.time(),
            "observation_age_s": None if observed_at is None else started - observed_at,
            "command_interval_s": None if previous_command is None else started - previous_command,
            # In prefetch mode the latest observation is newer than the one
            # used by the model. This is measured state at dispatch, not model-input age.
            "measured_latest": measured,
            "requested": requested,
            "sent": transmitted,
            "clipped_joints": [k for k, v in transmitted.items()
                               if k in requested and abs(v - requested[k]) > 1e-4],
        }
        stream.write(json.dumps(record) + "\n")
        previous_command = started
        if not frame_saved:
            from PIL import Image
            for name in ("cam0", "cam1"):
                if name in observation:
                    Image.fromarray(observation[name]).save(directory / f"{name}_first_action.png")
            frame_saved = True
        return sent

    robot_class.get_observation = observe
    robot_class.send_action = send

    def close():
        robot_class.get_observation = original_observe
        robot_class.send_action = original_send
        frame_writer.shutdown(wait=True)
        stream.close()

    return close


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from lerobot.rollout.interactive import InteractiveSession
    import commit_session
    commit_session.patch_session(InteractiveSession)
    from lerobot.robots.so_follower import SOFollower
    from lerobot.scripts.lerobot_rollout import main as rollout_main

    root = Path(__file__).resolve().parents[1]
    directory = root / "outputs/molmo_diagnostics" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    close = install_trace(SOFollower, directory)
    print(f"Diagnostic output: {directory}", flush=True)
    print("cam0 = top; cam1 = wrist. Joint targets and timestamped camera snapshots (up to 2 Hz) will be saved.", flush=True)
    print("In-session commit: /commit (down/clamp/up + hold) or /commit reset (then return to initial).", flush=True)
    try:
        rollout_main()
    finally:
        close()
        print(f"Diagnostic output saved: {directory}", flush=True)


if __name__ == "__main__":
    main()
