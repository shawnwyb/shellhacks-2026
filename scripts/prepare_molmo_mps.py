"""Prepare a local MolmoAct2 checkpoint for this LeRobot version and Mac MPS."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "molmoact2-mps"
BASE = ROOT / "models" / "molmoact2-upstream"


def migrate(config):
    config = dict(config)
    if config.pop("enable_lora_action_expert", False):
        raise ValueError("Action-expert LoRA requires a separate migration")
    lora = config.pop("enable_lora_vlm", False)
    frozen = config.pop("train_action_expert_only", False)
    config["train_mode_vlm"] = "lora" if lora else ("freeze" if frozen else "fft")
    config["dtype"] = config.pop("model_dtype", "float32")
    config.update(device="mps", enable_inference_cuda_graph=False, compile_model=False)
    config["checkpoint_path"] = str(BASE)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="Runtime precision; bfloat16 reduces memory use on the 48 GB Mac")
    args = parser.parse_args()
    revisions = {}
    repositories = (
        ("allenai/MolmoAct2-SO100_101", BASE),
        ("lerobot/MolmoAct2-SO100_101-LeRobot", DEST),
    )
    if not args.finalize_only:
        api = HfApi()
        for repo, dest in repositories:
            revision = api.model_info(repo).sha
            revisions[repo] = revision
            print(f"Downloading {repo} at {revision}", flush=True)
            snapshot_download(repo, revision=revision, local_dir=dest,
                              allow_patterns=["*.json", "*.safetensors", "*.jinja", "assets/*.png"],
                              max_workers=4)
    path = DEST / "config.json"
    backup = DEST / "config.original.json"
    original = json.loads((backup if backup.exists() else path).read_text())
    backup.write_text(json.dumps(original, indent=2) + "\n")
    config = migrate(original)
    config["dtype"] = args.dtype
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    import draccus
    draccus.decode(MolmoAct2Config, {k: v for k, v in config.items() if k != "type"})
    path.write_text(json.dumps(config, indent=2) + "\n")
    processor_path = DEST / "policy_preprocessor.json"
    processor = json.loads(processor_path.read_text())
    for step in processor["steps"]:
        if step["registry_name"] == "molmoact2_pack_inputs":
            step["config"]["checkpoint_path"] = str(BASE)
        elif step["registry_name"] == "device_processor":
            step["config"]["device"] = "mps"
    processor_path.write_text(json.dumps(processor, indent=2) + "\n")
    if revisions:
        (DEST / "source_revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
    print(f"Prepared and validated {DEST}", flush=True)


if __name__ == "__main__":
    main()
