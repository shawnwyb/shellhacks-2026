"""Live preview of both rollout cameras.

USB swapped: cam0 = top view (index 0), cam1 = wrist (index 1).
Press q to quit. No robot connection needed.

  python scripts/preview_cameras.py
  python scripts/preview_cameras.py --exposure -4 --brightness 100
  python scripts/preview_cameras.py --cam0 0 --cam1 1 --width 640 --height 480
"""
import argparse

import cv2


def open_capture(index, width, height):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def apply_manual(cap, args):
    import cv2
    if args.exposure is not None:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(args.brightness))
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(args.gain))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", type=int, default=0)
    ap.add_argument("--cam1", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure, e.g. -6 to -2 to tame blowout")
    ap.add_argument("--brightness", type=float, default=None, help="manual brightness 0-255")
    ap.add_argument("--gain", type=float, default=None, help="manual gain, lower tames blowout")
    args = ap.parse_args()

    cap0 = open_capture(args.cam0, args.width, args.height)
    cap1 = open_capture(args.cam1, args.width, args.height)
    if not cap0.isOpened() and not cap1.isOpened():
        raise SystemExit(f"could not open cam0 index {args.cam0} nor cam1 index {args.cam1}")
    if cap0.isOpened():
        apply_manual(cap0, args)
    if cap1.isOpened():
        apply_manual(cap1, args)

    print("cam0 = top view, cam1 = wrist. Press q to quit.", flush=True)
    try:
        while True:
            ok0, f0 = cap0.read() if cap0.isOpened() else (False, None)
            ok1, f1 = cap1.read() if cap1.isOpened() else (False, None)
            if not ok0 and not ok1:
                print("lost both camera streams", flush=True)
                break
            if ok0:
                cv2.imshow("cam0 top", f0)
            if ok1:
                cv2.imshow("cam1 wrist", f1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
