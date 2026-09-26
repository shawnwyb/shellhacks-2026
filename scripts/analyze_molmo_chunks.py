"""Replay saved observations on MPS; never import or connect to robot hardware."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"{name}.pos" for name in
          ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def matched_samples(directory):
    records = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    records = [r for r in records if r["observation_age_s"] is not None]
    observed_times = np.array([r["time"] - r["observation_age_s"] for r in records])
    matches = []
    for scene in sorted((directory / "frames").glob("*_cam0.jpg")):
        stamp = scene.name.split("_")[0]
        wrist = scene.with_name(f"{stamp}_cam1.jpg")
        if not wrist.exists():
            continue
        captured = int(stamp) / 1e9
        index = int(np.argmin(np.abs(observed_times - captured)))
        error = abs(observed_times[index] - captured)
        if error <= 0.05:
            matches.append((scene, wrist, records[index], error))
    if len(matches) < 4:
        raise RuntimeError("Need at least four camera/state pairs matched within 50 ms")
    # Skip the initial approach and sample throughout the later stalled trials.
    return [matches[i] for i in np.linspace(len(matches) // 3, len(matches) - 1, 4, dtype=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--task", default="Pick up the red tape roll.")
    args = parser.parse_args()
    directory = args.run or sorted((ROOT / "outputs/molmo_diagnostics").glob("*/actions.jsonl"))[-1].parent
    samples = matched_samples(directory)
    destination = directory / "offline_chunk_analysis.json"
    checkpoint = ROOT / "models/molmoact2-mps"
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.n_action_steps = 30
    config.rtc_config = None
    print("Loading MPS policy for saved-frame analysis; no hardware access.", flush=True)
    policy = MolmoAct2Policy.from_pretrained(checkpoint, config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "mps"}},
    )
    report = {"task": args.task, "joint_order": JOINTS, "samples": [],
              "limitations": "Approximate timestamp pairing, lossy JPEG snapshots, new random predictions; not exact replay or a physical grasp test. Joint targets alone do not establish fingertip height."}
    with torch.inference_mode():
        for scene, wrist, record, error in samples:
            state = [record["measured_latest"][k] for k in JOINTS]
            observation = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": args.task}
            for name, path in (("cam0", scene), ("cam1", wrist)):
                rgb = np.array(Image.open(path).convert("RGB"))
                observation[f"observation.images.{name}"] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            result = {"scene": str(scene), "wrist": str(wrist), "pairing_error_s": error,
                      "measured_state": state, "trials": []}
            for seed in (0, 1, 2):
                policy.reset()
                pre.reset()
                post.reset()
                torch.manual_seed(seed)
                torch.mps.manual_seed(seed)
                batch = pre(observation)
                started = time.monotonic()
                actions = post(policy.predict_action_chunk(batch)).squeeze(0).cpu().numpy()
                assert actions.shape == (30, 6) and np.isfinite(actions).all()
                summary = {"seed": seed, "seconds": time.monotonic() - started,
                           "first8_gripper_min": float(actions[:8, 5].min()),
                           "tail22_gripper_min": float(actions[8:, 5].min()),
                           "gripper_last": float(actions[-1, 5]),
                           "joint_change_after_action8": (actions[-1] - actions[7]).tolist(),
                           "actions": actions.tolist()}
                result["trials"].append(summary)
                print(scene.stem, {k: v for k, v in summary.items() if k != "actions"}, flush=True)
            report["samples"].append(result)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
