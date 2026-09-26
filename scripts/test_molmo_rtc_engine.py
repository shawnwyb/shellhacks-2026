"""Measure the real background inference engine with fixed inputs and no hardware."""
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from test_molmo_mps import BASE, DEST
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def main():
    config = PreTrainedConfig.from_pretrained(DEST)
    config.pretrained_path = str(DEST)
    config.rtc_config = RTCConfig(execution_horizon=30)
    print("Loading policy for background-engine test; no hardware is used.", flush=True)
    policy = MolmoAct2Policy.from_pretrained(DEST, config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(DEST),
        preprocessor_overrides={"device_processor": {"device": "mps"}},
    )
    joints = [f"{name}.pos" for name in
              ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
    old_state = np.array([-0.52734375, 189.140625, 181.40625,
                          60.64453125, -3.603515625, 1.0971786975860596])
    state = (old_state - np.array(config.joint_offsets)) / np.array(config.joint_signs)
    observation = dict(zip(joints, state.tolist(), strict=True))
    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": joints}}
    for key, name in (("cam0", "top"), ("cam1", "side")):
        observation[key] = np.array(Image.open(BASE / "assets" / f"sample_realsense_{name}_rgb.png").convert("RGB"))
        features[f"observation.images.{key}"] = {"dtype": "image", "shape": observation[key].shape}
    results = []
    for name, guided, steps in (("prefetch_10_steps", False, 10), ("guided_5_steps", True, 5)):
        config.rtc_config = RTCConfig(enabled=guided, execution_horizon=30)
        config.num_inference_steps = steps
        policy.init_rtc_processor()
        policy.reset()
        pre.reset()
        post.reset()
        results.append(benchmark(policy, pre, post, config, observation, features, name))
    (DEST / "rtc_engine_comparison.json").write_text(json.dumps(results, indent=2) + "\n")


def benchmark(policy, pre, post, config, observation, features, name):
    print(f"Testing {name}", flush=True)
    engine = RTCInferenceEngine(
        policy=policy, preprocessor=pre, postprocessor=post,
        robot_wrapper=SimpleNamespace(robot_type="so_follower"),
        rtc_config=config.rtc_config, dataset_features=features,
        task="Pick up the wrapped fortune cookie.", fps=30, device="mps",
        rtc_queue_threshold=30,
    )
    engine.start()
    engine.resume()
    started = time.monotonic()
    first = None
    received, missed = [], 0
    deadline = started
    try:
        while time.monotonic() - (first if first is not None else started) < (20 if first is not None else 60):
            engine.notify_observation(observation)
            action = engine.get_action(None)
            now = time.monotonic()
            if engine.failed:
                raise RuntimeError(engine.failure_traceback)
            if action is not None:
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite action")
                if first is None:
                    first = now
                    print(f"First action after {first - started:.2f}s; measuring 20 seconds.", flush=True)
                received.append(now)
            elif first is not None:
                missed += 1
            deadline += 1 / 30
            time.sleep(max(0, deadline - time.monotonic()))
        if len(received) < 2:
            raise RuntimeError("No sustained action output")
        result = {
            "mode": name,
            "actions": len(received), "empty_queue_ticks": missed,
            "effective_action_hz": (len(received) - 1) / (received[-1] - received[0]),
            "max_action_gap_s": max(b - a for a, b in zip(received, received[1:])),
            "note": "Fixed sample inputs; no camera, servo, or grasp test.",
        }
        print(json.dumps(result, indent=2), flush=True)
        (DEST / f"rtc_engine_{name}_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
