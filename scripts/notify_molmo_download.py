"""Wait for the active download and send a local macOS completion notification."""
import argparse
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "models/molmoact2-mps/source_revisions.json"


def notify(message):
    # Messages are fixed strings below; no external text is used as AppleScript.
    script = f'display notification "{message}" with title "MolmoAct2 download" sound name "Glass"'
    subprocess.run(["/usr/bin/osascript", "-e", script], check=True)
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    print(f"Watching download process {args.pid}", flush=True)
    while True:
        if MARKER.exists():
            notify("Download finished. Return to Codex to finish the MPS inference test.")
            return
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            notify("Download stopped before completion. Return to Codex to check the error.")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
