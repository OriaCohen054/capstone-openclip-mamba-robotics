from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
UI_LOG_DIR = PROJECT_ROOT / "outputs" / "ui_runs"
EMBEDDING_FILENAME = "openclip_vitb32_laion2b_s34b_b79k_embeddings.npy"
EMBEDDING_METADATA_FILENAME = "openclip_vitb32_laion2b_s34b_b79k_embeddings_metadata.json"
DEFAULT_DATASET_ROOT = "/home/linuxu/Downloads/scripted_6_18/scripted_raw"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "src" / "robotics_data_prep" / "best.pt"
DEFAULT_FILTER_CSV_NAME = "final_mamba_dataset.csv"
REQUIRED_FILTER_COLUMNS = {"Full_Path", "Status"}


@dataclass
class CsvSummary:
    path: Optional[str] = None
    exists: bool = False
    valid_schema: bool = False
    rows: int = 0
    success: int = 0
    failure: int = 0
    skipped: int = 0
    error: Optional[str] = None



def _format_local_timestamp(timestamp: float) -> str:
    """Format a filesystem timestamp using the server's local timezone."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def file_modified_text(path_value: Optional[str]) -> Optional[str]:
    """Return a human-readable modification time for an artifact file."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        return _format_local_timestamp(path.stat().st_mtime)
    except OSError:
        return None


def latest_embedding_modified_text(dataset_root: str) -> Optional[str]:
    """Return the newest OpenCLIP-cache timestamp below a dataset folder."""
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return None
    newest: Optional[float] = None
    try:
        for path in root.rglob(EMBEDDING_FILENAME):
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            newest = stamp if newest is None else max(newest, stamp)
    except OSError:
        return None
    return _format_local_timestamp(newest) if newest is not None else None


def process_started_text(value: Optional[str]) -> Optional[str]:
    """Convert the persisted UI process timestamp into a readable date."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value)

def browse_server_folder(initial_dir: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Open a native directory picker on the machine running Streamlit.

    This is suitable when Streamlit runs locally on the same Ubuntu desktop.
    When the server is headless or accessed remotely, the caller should expose
    a manual-path fallback.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        start = Path(initial_dir or DEFAULT_DATASET_ROOT).expanduser()
        if not start.exists():
            start = Path.home()
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(start),
            title="Choose RoboMamba raw trajectory folder",
            mustexist=True,
        )
        root.destroy()
        return (str(Path(selected).expanduser().resolve()), None) if selected else (None, None)
    except Exception as exc:  # Tk is optional on headless servers.
        return None, (
            "The native folder picker could not be opened on this machine. "
            f"Use the manual-path fallback instead. Details: {exc}"
        )


def count_trajectory_folders(dataset_root: str) -> int:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return 0
    try:
        return sum(1 for item in root.rglob("traj*") if item.is_dir())
    except OSError:
        return 0


def normalize_dataset_root(path_value: str) -> str:
    """Prefer the canonical scripted_raw subfolder when a parent folder is selected."""
    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_dir():
        return str(path)
    scripted_raw = path / "scripted_raw"
    if scripted_raw.is_dir() and count_trajectory_folders(str(scripted_raw)) > 0:
        return str(scripted_raw.resolve())
    return str(path.resolve())


def validate_dataset_root(path_value: str) -> tuple[bool, str]:
    if not path_value.strip():
        return False, "Choose a raw trajectory folder first."
    path = Path(path_value).expanduser()
    if not path.exists():
        return False, "The selected folder does not exist on the server."
    if not path.is_dir():
        return False, "The selected path is not a folder."
    trajectory_count = count_trajectory_folders(str(path))
    if trajectory_count == 0:
        return False, "The selected folder does not contain trajectory folders whose names start with 'traj'."
    return True, f"Dataset detected: {trajectory_count:,} trajectory folders."


def validate_filtered_csv(csv_path: Optional[str]) -> tuple[bool, str]:
    """Accept only a YOLO-filter CSV with the columns used by the pipeline."""
    if not csv_path:
        return False, "No filtered CSV was found."
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return False, "The filtered CSV does not exist."
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return False, f"The filtered CSV could not be read: {exc}"
    missing = sorted(REQUIRED_FILTER_COLUMNS.difference(header.columns))
    if missing:
        return False, "The CSV is not compatible with RoboMamba. Missing columns: " + ", ".join(missing)
    return True, "Compatible filtered CSV detected."


def find_filtered_csv(dataset_root: str) -> Optional[str]:
    """
    Return only the canonical CSV created by this project.

    We intentionally do not fall back to an arbitrary CSV in the selected
    folder, because an unrelated CSV may have an incompatible schema.
    """
    root = Path(dataset_root).expanduser()
    candidate = root / DEFAULT_FILTER_CSV_NAME
    valid, _ = validate_filtered_csv(str(candidate))
    return str(candidate) if valid else None


def read_csv_summary(csv_path: Optional[str]) -> CsvSummary:
    if not csv_path:
        return CsvSummary()
    path = Path(csv_path)
    if not path.exists():
        return CsvSummary(path=str(path))
    schema_ok, schema_message = validate_filtered_csv(str(path))
    if not schema_ok:
        return CsvSummary(path=str(path), exists=True, valid_schema=False, error=schema_message)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return CsvSummary(path=str(path), exists=True, error=str(exc))
    values = df["Status"].astype(str).str.upper()
    return CsvSummary(
        path=str(path),
        exists=True,
        valid_schema=True,
        rows=int(len(df)),
        success=int((values == "SUCCESS").sum()),
        failure=int((values == "FAILURE").sum()),
        skipped=int((values == "SKIPPED").sum()),
    )


def count_embeddings(dataset_root: str) -> int:
    root = Path(dataset_root).expanduser()
    return len(list(root.rglob(EMBEDDING_FILENAME))) if root.exists() else 0


def clear_embedding_cache(dataset_root: str) -> int:
    """Delete the OpenCLIP cache created by this project and return the number of removed files.

    The UI calls this only after the user explicitly chooses "Recompute all".
    Starting from an empty cache makes an interrupted recomputation resumable:
    the regular OpenCLIP script can skip trajectories already regenerated and
    continue with the remaining ones.
    """
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return 0
    removed = 0
    for filename in (EMBEDDING_FILENAME, EMBEDDING_METADATA_FILENAME):
        for path in root.rglob(filename):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def discover_checkpoints() -> dict[str, dict[str, Any]]:
    """Discover checkpoint folders and expose safe UI capabilities.

    A last_checkpoint.pth file exists after every completed epoch, including
    successful runs.  Therefore its existence alone is not enough to show a
    Resume button.  A run is resumable only when a resume checkpoint exists
    and the final metrics file has not been written yet.
    """
    if not CHECKPOINTS_DIR.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for run_dir in sorted((path for path in CHECKPOINTS_DIR.iterdir() if path.is_dir()), key=lambda p: p.name):
        model_path = run_dir / "model.pth"
        resume_path = run_dir / "last_checkpoint.pth"
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not model_path.exists() and not resume_path.exists():
            continue
        metrics_payload = load_json(str(metrics_path)) if metrics_path.exists() else {}
        training_completed = metrics_path.exists()
        can_resume = resume_path.exists() and not training_completed
        result[run_dir.name] = {
            "run_name": run_dir.name,
            "checkpoint_dir": str(run_dir),
            "weights_path": str(model_path),
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "resume_path": str(resume_path),
            "has_model": model_path.exists(),
            "has_resume": resume_path.exists(),
            "can_resume": can_resume,
            "training_completed": training_completed,
            "metrics": metrics_payload,
            "last_updated": file_modified_text(str(metrics_path if metrics_path.exists() else resume_path if resume_path.exists() else model_path)),
        }
    return result


def load_json(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalise_visible_config_value(value: Any) -> Any:
    """Normalise the user-editable training fields for a safe UI pre-flight comparison."""
    if isinstance(value, float):
        return round(value, 10)
    return value


def find_visible_training_match(checkpoints: dict[str, dict[str, Any]], config: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return an existing trained checkpoint with the same visible new-run settings.

    The authoritative duplicate check still lives in src/train.py and uses the full
    registry hash.  This lightweight UI check catches the common case before a GPU
    process is launched, so the user gets a clear choice instead of a failed-looking
    training screen.
    """
    keys = ("seq_length", "d_model", "batch_size", "learning_rate", "max_epochs", "patience")
    target = {key: _normalise_visible_config_value(config.get(key)) for key in keys}
    for item in checkpoints.values():
        if not item.get("has_model"):
            continue
        saved = load_json(item.get("config_path", ""))
        candidate = {key: _normalise_visible_config_value(saved.get(key)) for key in keys}
        if candidate == target:
            return item
    return None


def shorten_middle(value: Any, max_length: int = 28) -> str:
    """Return a compact display label while preserving the full value elsewhere."""
    text = str(value or "--")
    if len(text) <= max_length:
        return text
    keep = max(4, (max_length - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def duplicate_checkpoint_from_lines(lines: list[str]) -> Optional[str]:
    """Extract a duplicate-run name from the authoritative train.py message."""
    text = "\n".join(lines)
    match = re.search(r"A checkpoint with the same configuration already exists\.\s*\nRun name:\s*(.+)", text)
    return match.group(1).strip() if match else None


def latest_stage_log(stage: str) -> Optional[str]:
    if not UI_LOG_DIR.exists():
        return None
    matches = sorted(UI_LOG_DIR.glob(f"{stage}_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else None


def read_log_lines(log_path: Optional[str], max_lines: int = 400) -> list[str]:
    if not log_path:
        return []
    path = Path(log_path)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]


def build_yolo_command(dataset_root: str, weights_path: str, restart: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(SRC_DIR / "robotics_data_prep" / "filter_data.py"),
        "--data_dir", dataset_root,
        "--weights", weights_path,
        "--output", DEFAULT_FILTER_CSV_NAME,
    ]
    if restart:
        cmd.append("--restart")
    return cmd


def build_openclip_command(csv_path: str, batch_size: int = 32, overwrite: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(SRC_DIR / "openclip_embed.py"),
        "--csv_path", csv_path,
        "--status_filter", "SUCCESS",
        "--batch_size", str(batch_size),
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def build_train_command(config: dict[str, Any], resume_path: Optional[str] = None, force_train: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(SRC_DIR / "train.py"),
        "--filtered_csv", str(config["filtered_csv"]),
        "--status_filter", "SUCCESS",
        "--seq_length", str(config["seq_length"]),
        "--d_model", str(config["d_model"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--max_epochs", str(config["max_epochs"]),
        "--patience", str(config["patience"]),
    ]
    run_name = str(config.get("run_name") or "").strip()
    if run_name:
        cmd += ["--run_name", run_name]
    if force_train:
        cmd.append("--force_train")
    if resume_path:
        cmd += ["--resume_checkpoint", resume_path]
    return cmd



def checkpoint_max_preview_examples(checkpoint: Optional[dict[str, Any]], default_limit: int = 50) -> int:
    """Return the largest useful number of prediction previews for a checkpoint.

    Completed training runs persist ``num_val_windows`` in metrics.json.  That
    value is the exact number of validation samples available to Evaluation,
    and therefore the true upper bound for preview examples.  Older checkpoints
    may not contain that field, so we fall back to a conservative limit instead
    of blocking Evaluation.
    """
    if not checkpoint:
        return max(1, int(default_limit))
    metrics = checkpoint.get("metrics") or {}
    candidates = [
        metrics.get("num_val_windows"),
        (metrics.get("dataset_stats") or {}).get("num_val_windows"),
    ]
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return max(1, int(default_limit))

def build_evaluate_command(checkpoint: dict[str, Any], csv_path: str, mode: str, examples: int) -> list[str]:
    # Clamp at the service boundary as well as in the UI.  The upper bound is
    # checkpoint-specific: prediction previews cannot exceed the number of
    # available validation windows saved with the selected trained weights.
    max_examples = checkpoint_max_preview_examples(checkpoint)
    examples = max(1, min(max_examples, int(examples)))
    cmd = [
        sys.executable,
        str(SRC_DIR / "evaluate.py"),
        "--checkpoint_dir", str(checkpoint["checkpoint_dir"]),
        "--filtered_csv", csv_path,
        "--status_filter", "SUCCESS",
        "--show_examples", str(examples),
    ]
    if mode == "Quick":
        cmd += ["--max_batches", "10"]
    return cmd


def parse_yolo_progress(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    matches = re.findall(r"\[PROGRESS\]\s+Saved\s+(\d+)/(\d+)\s+trajectories", text)
    done, total = (int(matches[-1][0]), int(matches[-1][1])) if matches else (0, 0)
    return {"done": done, "total": total, "ratio": done / total if total else 0.0}


def parse_embedding_progress(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    found = re.findall(r"Extracting OpenCLIP embeddings:\s*(\d+)%.*?\|\s*(\d+)/(\d+)", text)
    if found:
        percent, done, total = found[-1]
        return {"done": int(done), "total": int(total), "ratio": int(percent) / 100.0}
    created = re.findall(r"Created:\s*(\d+)", text)
    return {"done": int(created[-1]) if created else 0, "total": 0, "ratio": 0.0}


def parse_training_progress(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    found = re.findall(
        r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Val Loss:\s*([0-9.eE+-]+)\s*\|\s*(new_best|no_improvement)",
        text,
    )
    structured = re.findall(
        r"\[TRAIN_PROGRESS\]\s+Epoch\s+(\d+)/(\d+)\s*\|\s*Patience:\s*(\d+)/(\d+)\s*\|\s*Best epoch:\s*(\d+)\s*\|\s*Best val loss:\s*([0-9.eE+-]+)",
        text,
    )
    early = re.findall(r"Early stopping triggered at epoch\s+(\d+)", text)
    best_epoch = re.findall(r"Best epoch:\s*(\d+)", text)
    best_val = re.findall(r"Best validation loss:\s*([0-9.eE+-]+)", text)
    if not found:
        return {
            "epoch": 0, "max_epochs": 0, "ratio": 0.0,
            "train_loss": None, "val_loss": None, "history": [],
            "patience_counter": 0, "patience": 0, "best_epoch": None,
            "best_val_loss": None, "early_stopping": False,
        }
    structured_by_epoch = {
        int(epoch): {
            "patience_counter": int(pc), "patience": int(p),
            "best_epoch": int(be), "best_val_loss": float(bv),
        }
        for epoch, _max_epochs, pc, p, be, bv in structured
    }
    history = []
    for epoch, max_epochs, train, val, status in found:
        item = {
            "epoch": int(epoch), "max_epochs": int(max_epochs),
            "train_loss": float(train), "val_loss": float(val), "status": status,
        }
        item.update(structured_by_epoch.get(int(epoch), {}))
        history.append(item)
    current = history[-1]
    patience_counter = 0
    patience = 0
    parsed_best_epoch = int(best_epoch[-1]) if best_epoch else None
    parsed_best_val = float(best_val[-1]) if best_val else None
    if structured:
        _, _, pc, p, be, bv = structured[-1]
        patience_counter, patience = int(pc), int(p)
        parsed_best_epoch, parsed_best_val = int(be), float(bv)
    return {
        **current,
        "ratio": current["epoch"] / current["max_epochs"] if current["max_epochs"] else 0.0,
        "history": history,
        "patience_counter": patience_counter,
        "patience": patience,
        "best_epoch": parsed_best_epoch,
        "best_val_loss": parsed_best_val,
        "early_stopping": bool(early),
    }


def parse_evaluation_metrics(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)

    def last(pattern: str, cast=float):
        items = re.findall(pattern, text)
        return cast(items[-1]) if items else None

    per_dim = re.findall(r"Per-dimension MAE:\s*\[([^\]]+)\]", text)
    examples = re.findall(r"Example\s+(\d+):", text)
    progress = re.findall(
        r"\[EVAL_PROGRESS\]\s+Batches\s+(\d+)/(\d+)\s*\|\s*Samples:\s*(\d+)\s*\|\s*MSE:\s*([0-9.eE+-]+)\s*\|\s*MAE:\s*([0-9.eE+-]+)",
        text,
    )
    progress_payload = {"done_batches": 0, "total_batches": 0, "processed_samples": 0, "running_mse": None, "running_mae": None, "ratio": 0.0}
    if progress:
        done, total, samples, mse, mae = progress[-1]
        progress_payload = {
            "done_batches": int(done), "total_batches": int(total),
            "processed_samples": int(samples), "running_mse": float(mse),
            "running_mae": float(mae), "ratio": int(done) / int(total) if int(total) else 0.0,
        }

    # Prefer structured JSON examples printed by evaluate.py v36+.
    # They can include dataset_index, trajectory_id, frame_index and frame_path.
    prediction_examples = []
    for line in lines:
        if "[EVAL_EXAMPLE_JSON]" not in line:
            continue
        try:
            payload_text = line.split("[EVAL_EXAMPLE_JSON]", 1)[1].strip()
            payload = json.loads(payload_text)
            prediction_examples.append({
                "index": int(payload.get("example", len(prediction_examples) + 1)),
                "ground_truth": payload.get("ground_truth"),
                "prediction": payload.get("prediction"),
                "mae": payload.get("mae"),
                "dataset_index": payload.get("dataset_index"),
                "trajectory_id": payload.get("trajectory_id"),
                "frame_index": payload.get("frame_index"),
                "frame_path": payload.get("frame_path"),
            })
        except Exception:
            continue

    # Fallback for older logs that did not include structured JSON.
    if not prediction_examples:
        example_pattern = re.compile(
            r"Example\s+(\d+):\s*\n"
            r"Ground truth action:\s*\[([^\]]+)\]\s*\n"
            r"Predicted action:\s*\[([^\]]+)\]\s*\n"
            r"MAE:\s*([0-9.eE+-]+)",
            re.MULTILINE,
        )
        for example_id, gt, pred, mae in example_pattern.findall(text):
            prediction_examples.append({
                "index": int(example_id),
                "ground_truth": gt,
                "prediction": pred,
                "mae": float(mae),
            })

    # Fallback for very old logs that did not include per-example MAE.
    if not prediction_examples:
        gt_matches = re.findall(r"Ground truth action:\s*\[([^\]]+)\]", text)
        pred_matches = re.findall(r"Predicted action:\s*\[([^\]]+)\]", text)
        for idx, (gt, pred) in enumerate(zip(gt_matches, pred_matches), start=1):
            prediction_examples.append({"index": idx, "ground_truth": gt, "prediction": pred, "mae": None})

    saved_count_match = re.findall(r"\[EVAL_EXAMPLES_SAVED\]\s*(\d+)", text)
    saved_examples = int(saved_count_match[-1]) if saved_count_match else len(prediction_examples)

    final_samples = last(r"Evaluated samples:\s*(\d+)", int)
    final_mse = last(r"(?m)^MSE:\s*([0-9.eE+-]+)", float)
    final_mae = last(r"(?m)^MAE:\s*([0-9.eE+-]+)", float)
    return {
        "samples": final_samples or progress_payload["processed_samples"],
        "mse": final_mse if final_mse is not None else progress_payload["running_mse"],
        "mae": final_mae if final_mae is not None else progress_payload["running_mae"],
        "per_dimension_mae": per_dim[-1] if per_dim else None,
        "printed_examples": saved_examples if prediction_examples else len(examples),
        "prediction_examples": prediction_examples,
        **progress_payload,
    }

def process_completed_successfully(stage: str, lines: list[str]) -> bool:
    text = "\n".join(lines)
    markers = {
        "yolo": "Filtering completed successfully.",
        "openclip": "Embedding extraction complete.",
        "training": "Training complete.",
        "evaluation": "Evaluation complete.",
    }
    return markers.get(stage, "") in text


def suggest_run_name(config: dict[str, Any]) -> str:
    lr = str(config.get("learning_rate", 0.001)).replace(".", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mamba_k{int(config.get('seq_length', 10))}_d{int(config.get('d_model', 128))}_lr{lr}_{stamp}"


def artifact_state(dataset_root: str) -> dict[str, Any]:
    valid, message = validate_dataset_root(dataset_root)
    csv_path = find_filtered_csv(dataset_root) if valid else None
    summary = read_csv_summary(csv_path)
    embeddings = count_embeddings(dataset_root) if valid else 0
    checkpoints = discover_checkpoints()
    eval_lines = read_log_lines(latest_stage_log("evaluation"))
    return {
        "dataset_valid": valid,
        "dataset_message": message,
        "filtered_csv": csv_path,
        "csv_summary": summary,
        "embedding_count": embeddings,
        "trajectory_count": count_trajectory_folders(dataset_root) if valid else 0,
        "openclip_cache_complete": bool(summary.success > 0 and embeddings >= summary.success),
        "checkpoints": checkpoints,
        "latest_eval_metrics": parse_evaluation_metrics(eval_lines) if process_completed_successfully("evaluation", eval_lines) else {},
    }
