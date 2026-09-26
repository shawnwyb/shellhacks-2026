"""Test the full saved policy pipeline on bundled sample images, without hardware."""
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
DEST = ROOT / "models" / "molmoact2-mps"
BASE = ROOT / "models" / "molmoact2-upstream"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtc", action="store_true", help="Test RTC with a leftover action prefix")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to this process")
    config = PreTrainedConfig.from_pretrained(DEST)
    config.pretrained_path = str(DEST)
    if args.rtc:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        config.rtc_config = RTCConfig(execution_horizon=30)
    print("Loading policy on MPS (no robot connection)...", flush=True)
    started = time.perf_counter()
    policy = MolmoAct2Policy.from_pretrained(DEST, config=config).eval()
    print(f"Loaded in {time.perf_counter() - started:.1f}s", flush=True)
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(DEST),
        preprocessor_overrides={"device_processor": {"device": "mps"}},
    )
    # Published sample state is in the model's older joint frame. Convert it
    # back to the robot frame so the saved preprocessor applies the correction once.
    old_state = torch.tensor([-0.52734375, 189.140625, 181.40625,
                              60.64453125, -3.603515625, 1.0971786975860596])
    state = (old_state - torch.tensor(config.joint_offsets)) / torch.tensor(config.joint_signs)
    observation = {"observation.state": state, "task": "Pick up the wrapped fortune cookie."}
    for key, name in (("cam0", "top"), ("cam1", "side")):
        picture = Image.open(BASE / "assets" / f"sample_realsense_{name}_rgb.png").convert("RGB")
        observation[f"observation.images.{key}"] = (
            torch.from_numpy(np.array(picture)).permute(2, 0, 1).float() / 255
        )
    # RTC guidance enables autograd locally; inference_mode would prevent that.
    with torch.no_grad():
        batch = pre(observation)
        previous = None
        results = []
        for trial in range(3 if args.rtc else 2):
            policy.reset()
            torch.mps.synchronize()
            started = time.perf_counter()
            kwargs = {}
            if args.rtc:
                kwargs = {"inference_delay": 10 if previous is not None else 0,
                          "prev_chunk_left_over": previous}
            chunk = policy.predict_action_chunk(batch, **kwargs)
            action = post(chunk[:, 0, :])
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            if tuple(chunk.shape) != (1, config.n_action_steps, 6):
                raise RuntimeError(f"Unexpected action shape: {chunk.shape}")
            if not torch.isfinite(chunk).all() or not torch.isfinite(action).all():
                raise RuntimeError("Non-finite model outputs")
            print(f"Trial {trial + 1}: {tuple(chunk.shape)}, finite outputs, {elapsed:.2f}s", flush=True)
            result = {"trial": trial + 1, "seconds": elapsed,
                      "shape": list(chunk.shape), "device": str(chunk.device),
                      "first_action": action.cpu().tolist(),
                      "note": "Bundled sample images only; not a grasp success test."}
            result["rtc"] = args.rtc
            results.append(result)
            filename = "mps_rtc_test_results.json" if args.rtc else "mps_test_result.json"
            (DEST / filename).write_text(json.dumps(results if args.rtc else result, indent=2) + "\n")
            previous = chunk.detach().squeeze(0)[10:].clone()
    print("PASS: model and saved processors completed MPS inference; no hardware accessed.")


if __name__ == "__main__":
    main()
