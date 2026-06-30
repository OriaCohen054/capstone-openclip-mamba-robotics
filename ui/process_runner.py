from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def runtime_file(project_root: Path) -> Path:
    return Path(project_root).resolve() / "outputs" / "ui_runtime" / "current_process.json"


def flow_file(project_root: Path) -> Path:
    return Path(project_root).resolve() / "outputs" / "ui_runtime" / "flow_state.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_current_process(project_root: Path) -> Optional[dict]:
    path = runtime_file(project_root)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def persist_process(project_root: Path, process_info: dict) -> None:
    _write_json(runtime_file(project_root), process_info)


def clear_current_process(project_root: Path) -> None:
    try:
        runtime_file(project_root).unlink(missing_ok=True)
    except OSError:
        pass


def load_flow_state(project_root: Path) -> dict:
    path = flow_file(project_root)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def persist_flow_state(project_root: Path, flow_state: dict) -> None:
    _write_json(flow_file(project_root), flow_state)


def clear_flow_state(project_root: Path) -> None:
    try:
        flow_file(project_root).unlink(missing_ok=True)
    except OSError:
        pass


def start_process(command: list[str], project_root: Path, stage: str, metadata: Optional[dict] = None) -> dict:
    project_root = Path(project_root).resolve()
    log_dir = project_root / "outputs" / "ui_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stage}_{timestamp}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
    process_info = {
        "pid": process.pid,
        "stage": stage,
        "command": command,
        "log_path": str(log_path),
        "started_at": timestamp,
        "requested_stop": None,
        "metadata": metadata or {},
    }
    persist_process(project_root, process_info)
    return process_info


def _linux_process_state(pid: int) -> str:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return ""
    try:
        content = stat_path.read_text(errors="ignore")
        return content.split(")", 1)[1].strip().split()[0]
    except Exception:
        return ""


def is_process_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return _linux_process_state(int(pid)) != "Z"


def stop_process(pid: Optional[int]) -> bool:
    if not pid or not is_process_running(pid):
        return False
    pid = int(pid)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except OSError:
            return False


def read_log_tail(log_path: Optional[str], max_lines: Optional[int] = 180) -> list[str]:
    if not log_path:
        return []
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines if max_lines is None else lines[-max_lines:]
    except OSError:
        return []


def joined_log(lines: Iterable[str]) -> str:
    return "\n".join(lines)
