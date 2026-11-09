"""Log the leg-stand carriage height from the camera tracker's event stream, on wall-clock time.

    python3 height_log.py --url http://localhost:8790 --seconds 60 --out heights.npz

Reads ``/events`` of ``testing/leg_stand/tag_height.py`` (branch ab3233/leg-stand-tag-height). Each
event carries ``t`` (capture time on the tracker's monotonic clock) and ``lat`` (capture to send, ms).
Capture wall time is estimated as receipt wall time - lat for every event, and the tracker clock's
wall offset is their median, so every sample is placed at its capture time even when processing
lags. Standalone (standard library and numpy); ``h`` is the tracker's number: the zeroed and scaled
carriage height in mm once zeroed, else the raw lens-to-plate distance (it falls as the carriage
rises).
"""

import argparse
import json
import time
import urllib.request

import numpy as np


def log(url, seconds):
    rows = []
    deadline = time.time() + seconds
    with urllib.request.urlopen(url.rstrip("/") + "/events", timeout=10) as stream:
        for raw in stream:
            now = time.time()
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            s = json.loads(line[5:])
            rows.append(
                (now, s["t"], s["lat"], np.nan if s["h"] is None else s["h"], bool(s["both"]))
            )
            if now > deadline:
                break
    data = np.array(rows, dtype=float)
    offset = float(np.median(data[:, 0] - data[:, 2] / 1000 - data[:, 1]))
    return {
        "wall": data[:, 1] + offset,
        "h": data[:, 3],
        "both": data[:, 4],
        "received_wall": data[:, 0],
        "tracker_clock_offset_s": offset,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8790")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--state", help="also save the tracker's /state (lock and calibration) here"
    )
    args = parser.parse_args(argv)
    if args.state:
        with urllib.request.urlopen(args.url.rstrip("/") + "/state", timeout=10) as r:
            open(args.state, "w").write(r.read().decode())
    out = log(args.url, args.seconds)
    np.savez_compressed(args.out, **out)
    good = np.isfinite(out["h"])
    print(
        f"{args.out}: {len(out['h'])} samples, {good.mean():.0%} with a height, "
        f"{len(out['h']) / max(1e-9, out['wall'][-1] - out['wall'][0]):.0f} Hz"
    )


if __name__ == "__main__":
    main()
