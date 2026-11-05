"""Manually resume the saved DiT run; --dry-run verifies without launching.

Uses only standard-library orchestration and the existing recorded trainer
command. Provenance: docs/provenance/dit_training_pause_20261101.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = root / "outputs/reactflow_dit"
    launch_path = base / "seed_2026.launch.json"
    runtime_path = base / "seed_2026.runtime.json"
    launch = json.loads(launch_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    pause = json.loads(Path(launch["pause_record"]).read_text())

    if launch["status"] != "paused_by_user" or runtime["status"] != "paused_by_user":
        raise RuntimeError("The run is not marked paused; inspect its current state first")
    for proc in pause["processes"]:
        stat = Path(f"/proc/{proc['pid']}/stat")
        if stat.exists() and stat.read_text().split(") ", 1)[1].split()[0] != "Z":
            raise RuntimeError(f"Recorded training PID {proc['pid']} is still active")

    source = Path(launch["cwd"])
    manifest = json.loads(Path(launch["source_manifest"]).read_text())
    checkpoint = Path(pause["checkpoint_path"])
    identities = [(source / name, digest) for name, digest in manifest["files"].items()]
    identities.append((checkpoint, pause["checkpoint_sha256"]))
    for path, expected in identities:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Saved source/checkpoint checksum differs: {path}")

    command = launch["base_command"] + ["--resume", str(checkpoint)]
    summary = {"checkpoint": str(checkpoint), "global_step": pause["checkpoint_global_step"],
               "cwd": str(source), "verified_files": len(identities)}
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "training_started": False, **summary}, indent=2))
        return

    env = os.environ.copy()
    for name in launch["environment_removed"]:
        env.pop(name, None)
    env.update(launch["environment_overrides"])
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    console = base / f"seed_2026.resume_{now.strftime('%Y%m%dT%H%M%S%f')}.log"
    with console.open("xb") as stream:
        process = subprocess.Popen(command, cwd=source, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    timestamp = now.isoformat()
    runtime.update({"status": "running", "stage": "resumed_full_training",
                    "torchrun_pid": process.pid, "process_group": process.pid,
                    "resumed_at": timestamp, "console_log": str(console),
                    "last_verified_log_step": pause["checkpoint_global_step"],
                    "auto_restart": False, "restart_disabled": False})
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    launch.update({"status": "resumed_training_running", "resume_torchrun_pid": process.pid,
                   "resumed_at": timestamp, "resume_console": str(console),
                   "resume_command": command, "auto_restart": False, "restart_disabled": False})
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps({"status": "launched", "torchrun_pid": process.pid,
                      "console_log": str(console), **summary}, indent=2))


if __name__ == "__main__":
    main()
