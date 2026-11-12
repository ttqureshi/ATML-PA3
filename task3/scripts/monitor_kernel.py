"""Poll a Kaggle kernel until it leaves the running/queued states."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def status(ref: str) -> str:
    out = subprocess.check_output(["kaggle", "kernels", "status", ref], text=True, stderr=subprocess.STDOUT)
    line = out.strip().splitlines()[-1]
    if "status" in line:
        return line.split("status", 1)[-1].strip().strip('"')
    return line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="mtayyabtahir/atml-pa3-t3-live")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-minutes", type=int, default=360)
    args = parser.parse_args()
    deadline = time.time() + args.max_minutes * 60
    last = None
    failures = 0
    while time.time() < deadline:
        try:
            current = status(args.ref)
            failures = 0
        except subprocess.CalledProcessError as exc:
            failures += 1
            print(time.strftime("%H:%M:%S"), "status_error", failures, (exc.output or "")[:200], flush=True)
            if failures >= 8:
                raise
            time.sleep(min(30 * failures, 120))
            continue
        if current != last:
            print(time.strftime("%H:%M:%S"), current, flush=True)
            last = current
        done_tokens = (
            "KernelWorkerStatus.COMPLETE",
            "KernelWorkerStatus.ERROR",
            "KernelWorkerStatus.CANCELLED",
            "KernelWorkerStatus.CANCEL",
        )
        if any(token in current for token in done_tokens):
            print("DONE", current, flush=True)
            sys.exit(0 if "COMPLETE" in current else 1)
        time.sleep(args.interval)
    raise SystemExit(f"Timed out waiting for {args.ref}; last={last}")


if __name__ == "__main__":
    main()
