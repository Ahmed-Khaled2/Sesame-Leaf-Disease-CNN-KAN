"""
run_queue.py - run a list of train.py commands, split across several terminals.

Each line of a phase file holds the arguments for ONE train.py run.
Open 3 terminals (or 3 VS Code sessions) and start one worker in each:

  python run_queue.py phases/phase1.txt --worker 0 --workers 3
  python run_queue.py phases/phase1.txt --worker 1 --workers 3
  python run_queue.py phases/phase1.txt --worker 2 --workers 3

Worker 0 takes lines 1, 4, 7 ..., worker 1 takes lines 2, 5, 8 ..., and so on.
A line that starts with @ runs another script instead of train.py, e.g.  @gradcam.py
(the script gets QUEUE_NAME / QUEUE_WORKER / QUEUE_WORKERS, so it can wait for the other workers).
Finished runs are skipped automatically, so a queue can simply be started again
after a crash or a restart. Do NOT start the same worker twice at the same time.
"""
import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase_file")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="only print the commands")
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.phase_file).read_text(encoding="utf-8").splitlines()]
    cmds = [l for l in lines if l and not l.startswith("#")]
    mine = [c for i, c in enumerate(cmds) if i % args.workers == args.worker]
    log_dir = HERE / "outputs" / "queue_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{Path(args.phase_file).stem}_worker{args.worker}.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== {dt.datetime.now():%Y-%m-%d %H:%M:%S} worker {args.worker}/{args.workers}: "
        f"{len(mine)} of {len(cmds)} runs in {args.phase_file}")
    failed = []
    env = dict(os.environ, QUEUE_NAME=Path(args.phase_file).stem, QUEUE_WORKER=str(args.worker),
               QUEUE_WORKERS=str(args.workers))
    for n, c in enumerate(mine, 1):
        if c.startswith("@"):
            parts = shlex.split(c[1:])
            full = [sys.executable, str(HERE / parts[0])] + parts[1:]
            label = c[1:]
        else:
            full = [sys.executable, str(HERE / "train.py")] + shlex.split(c)
            label = "train.py " + c
        log(f"[{n}/{len(mine)}] {dt.datetime.now():%H:%M:%S}  {label}")
        if args.dry:
            continue
        code = subprocess.call(full, cwd=str(HERE), env=env)
        if code != 0:
            failed.append(c)
            log(f"   !! exit code {code} - continuing with the next run")
    log(f"=== worker {args.worker} finished {dt.datetime.now():%Y-%m-%d %H:%M:%S}. "
        f"Failed runs: {len(failed)}")
    for c in failed:
        log(f"   FAILED: {c}")


if __name__ == "__main__":
    main()
