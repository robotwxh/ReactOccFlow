"""Launch the verified original CFM continuation; --dry-run only checks files.

Uses the exact archived trainer and its strict resume checks. Source/checkpoint
relocation is documented in docs/provenance/cfm_training_resume_20261103.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/reactflow_joint"
    launch_path = base / "seed_2026.resume_20261103.launch.json"
    with (base / "seed_2026.resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch = json.loads(launch_path.read_text())
        if launch["status"] not in ("prepared", "paused_by_user"):
            raise RuntimeError("This continuation has already launched; inspect its runtime record")
        source = Path(launch["cwd"])
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                command = (process / "cmdline").read_bytes().split(b"\0")
                if (b"scripts.train_react_flow" in command
                        and (process / "cwd").resolve() == source):
                    raise RuntimeError(f"CFM training is already active at PID {process.name}")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        manifest = json.loads(Path(launch["source_manifest"]).read_text())
        identities = [(source / name, expected) for name, expected in manifest["files"].items()]
        identities.append((Path(launch["checkpoint"]), launch["checkpoint_sha256"]))
        for path, expected in identities:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Expected an independent regular file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Saved source/checkpoint checksum differs: {path}")
        summary = {"backbone": launch["backbone"], "checkpoint_step": launch["global_step"],
                   "cwd": str(source), "verified_files": len(identities)}
        if args.dry_run:
            print(json.dumps({"status": "dry_run_passed", "training_started": False, **summary}, indent=2))
            return
        env = os.environ.copy()
        for name in launch["environment_removed"]:
            env.pop(name, None)
        env.update(launch["environment_overrides"])
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        console = base / f"seed_2026.cfm_resume_{now.strftime('%Y%m%dT%H%M%S%f')}.log"
        runtime_path = Path(launch["runtime_path"])
        runtime = json.loads(runtime_path.read_text())
        (base / "resume_20261103/runtime.before_resume.json").write_text(json.dumps(runtime, indent=2) + "\n")
        with console.open("xb") as stream:
            process = subprocess.Popen(launch["command"], cwd=source, env=env,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        for key in ("stopped_at", "reason", "stop_record", "latest_logged_step",
                    "new_training_started", "elapsed_seconds", "current_default_backbone"):
            runtime.pop(key, None)
        runtime.update({"status": "starting", "stage": "cfm_strict_resume",
                        "torchrun_pid": process.pid, "process_group": process.pid,
                        "command": launch["command"], "cwd": str(source),
                        "resumed_at": now.isoformat(), "console_log": str(console),
                        "saved_checkpoint_step": launch["global_step"],
                        "current_backbone": launch["backbone"],
                        "previous_stop_record": str(base / "seed_2026.stopped_by_user.json"),
                        "auto_restart": False, "restart_disabled": True,
                        "launcher_pid": None, "guardian_pid": None,
                        "latest_verified_training_step": None})
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
        launch.update({"status": "launched", "torchrun_pid": process.pid,
                       "launched_at": now.isoformat(), "console_log": str(console)})
        launch_path.write_text(json.dumps(launch, indent=2) + "\n")
        print(json.dumps({"status": "launched", "torchrun_pid": process.pid,
                          "console_log": str(console), **summary}, indent=2))


if __name__ == "__main__":
    main()
