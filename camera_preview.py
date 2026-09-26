"""Preview the wrist and scene cameras; measure delivered frames per second."""

import argparse
import threading
import time

import cv2


def capture(index, label, args, state, lock, stop):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {label} camera {index}")
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, args.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
            (cv2.CAP_PROP_FPS, args.fps),
        ):
            cap.set(prop, value)
        print(
            f"{label} [{index}]: requested {args.width}x{args.height} @ {args.fps} FPS; "
            f"backend reports {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
            f"{cap.get(cv2.CAP_PROP_FPS):.2f} FPS",
            flush=True,
        )
        warmup_end = time.perf_counter() + 3
        started = None
        count = 0
        report_start = None
        report_count = 0
        measured = None
        deadline = time.perf_counter() + args.seconds + 15
        while not stop.is_set():
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok:
                raise RuntimeError(f"{label}: camera read failed")
            if args.benchmark and now > deadline:
                raise RuntimeError(f"{label}: timed out before completing measurement")
            if now >= warmup_end:
                if started is None:
                    started = report_start = now
                else:
                    count += 1
                    report_count += 1
                    if now - report_start >= 5:
                        measured = report_count / (now - report_start)
                        print(
                            f"{label}: {measured:.1f} FPS over last "
                            f"{now - report_start:.1f}s; average "
                            f"{count / (now - started):.1f} FPS over {now - started:.1f}s",
                            flush=True,
                        )
                        report_start, report_count = now, 0
                    if args.benchmark and now - started >= args.seconds:
                        with lock:
                            state[label] = count / (now - started)
                        break
            if not args.benchmark:
                with lock:
                    state[label] = (frame, measured)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        stop.set()
    finally:
        cap.release()


def benchmark(args):
    results = []
    for name, cameras in (
        ("Both together", [(args.wrist, "Wrist"), (args.scene, "Scene")]),
        ("Wrist alone", [(args.wrist, "Wrist")]),
        ("Scene alone", [(args.scene, "Scene")]),
    ):
        print(f"\n{name}: 3s warmup, then {args.seconds:g}s measurement", flush=True)
        state, lock, stop = {}, threading.Lock(), threading.Event()
        workers = [threading.Thread(
            target=capture, args=(index, label, args, state, lock, stop), daemon=True
        ) for index, label in cameras]
        try:
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + args.seconds + 25
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
            if any(worker.is_alive() for worker in workers) or stop.is_set():
                raise RuntimeError("Camera test failed or stalled; stopping the benchmark")
            results.append((name, state))
        finally:
            stop.set()
        time.sleep(1)
    print("\nRESULTS — delivered FPS, no preview windows", flush=True)
    for name, state in results:
        print(f"{name}: " + ", ".join(f"{label} {fps:.2f} FPS" for label, fps in state.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--wrist", type=int, default=0)
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true",
                        help="Test both cameras, then each individually, without preview windows")
    parser.add_argument("--seconds", type=float, default=15,
                        help="Measurement duration per benchmark stage (default: 15)")
    args = parser.parse_args()
    if args.wrist == args.scene:
        parser.error("Wrist and scene must use different camera indices")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.benchmark:
        benchmark(args)
        return
    state, lock, stop = {}, threading.Lock(), threading.Event()
    workers = []
    cameras = [(args.wrist, "Wrist"), (args.scene, "Scene")]
    print("Focus a preview window and press Q or Escape to quit. Allow 20 seconds for FPS readings.")
    try:
        for index, label in cameras:
            cv2.namedWindow(label, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(label, args.width, args.height)
            worker = threading.Thread(
                target=capture, args=(index, label, args, state, lock, stop), daemon=True
            )
            workers.append(worker)
            worker.start()
        while not stop.is_set():
            with lock:
                snapshots = list(state.items())
            for label, (frame, fps) in snapshots:
                preview = frame.copy()
                status = "Measuring..." if fps is None else f"Delivered: {fps:.1f} FPS"
                cv2.putText(preview, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow(label, preview)
            if cv2.waitKey(10) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if any(cv2.getWindowProperty(label, cv2.WND_PROP_VISIBLE) < 1
                   for _, label in cameras):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=3)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
