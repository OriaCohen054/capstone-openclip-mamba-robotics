from __future__ import annotations

from pathlib import Path
from typing import Any

import base64
import html
import mimetypes

import streamlit as st

from components import (
    STAGES,
    brand,
    config_grid,
    empty_chart,
    log_box,
    metrics,
    path_box,
    per_dimension_mae_chart,
    prediction_comparison_chart,
    progress_card,
    robot_panel,
    section_card,
    simple_bars,
    source_config_block,
    stage_header,
    stage_stepper,
    training_loss_chart,
    training_status_chart,
)
from pipeline_services import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_YOLO_WEIGHTS,
    PROJECT_ROOT,
    artifact_state,
    browse_server_folder,
    build_evaluate_command,
    build_openclip_command,
    build_train_command,
    build_yolo_command,
    clear_embedding_cache,
    checkpoint_max_preview_examples,
    count_embeddings,
    count_trajectory_folders,
    discover_checkpoints,
    find_filtered_csv,
    file_modified_text,
    find_visible_training_match,
    latest_embedding_modified_text,
    load_json,
    normalize_dataset_root,
    parse_embedding_progress,
    parse_evaluation_metrics,
    parse_training_progress,
    parse_yolo_progress,
    process_started_text,
    shorten_middle,
    duplicate_checkpoint_from_lines,
    process_completed_successfully,
    read_csv_summary,
    suggest_run_name,
    validate_dataset_root,
)
from process_runner import (
    clear_current_process,
    clear_flow_state,
    is_process_running,
    load_current_process,
    load_flow_state,
    persist_flow_state,
    persist_process,
    read_log_tail,
    start_process,
    stop_process,
)
from ui_styles import apply_theme, inject_v32_result_css

st.set_page_config(page_title="RoboMamba", page_icon="🤖", layout="wide")
apply_theme()
inject_v32_result_css()

STAGE_KEY_TO_LABEL = {
    "yolo": "YOLO",
    "openclip": "OpenCLIP",
    "training": "Training",
    "evaluation": "Evaluation",
}

DEFAULT_TRAINING_CONFIG = {
    "seq_length": 10,
    "d_model": 128,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_epochs": 20,
    "patience": 5,
    "run_name": None,
}


def _initial_stage_states() -> dict[str, str]:
    return {stage: "not_started" for stage in STAGES}


def init_state() -> None:
    stored = load_flow_state(PROJECT_ROOT)
    defaults: dict[str, Any] = {
        "stage": "Dataset",
        "dataset_root": "",
        "dataset_valid": False,
        "dataset_message": "Choose a raw trajectory folder first.",
        "filtered_csv": None,
        "selected_checkpoint": None,
        "current_training_run": None,
        "last_used_checkpoint": None,
        "process": load_current_process(PROJECT_ROOT),
        "last_log_paths": {},
        "stage_states": _initial_stage_states(),
        "latest_eval_metrics": {},
        "examples_to_display": 5,
        "evaluation_mode": "Full",
        "yolo_weights": str(DEFAULT_YOLO_WEIGHTS),
        "openclip_batch_size": 32,
        "openclip_recompute_active": False,
        "training_config": dict(DEFAULT_TRAINING_CONFIG),
        "training_mode": "new",
        "resume_training_config": {},
        "force_train_new_run": False,
        "new_run_config_source_checkpoint": None,
        "folder_picker_error": None,
        "checkpoint_selector_sync_pending": False,
        "duplicate_checkbox_nonce": 0,
    }
    persistable_keys = {
        "stage",
        "dataset_root",
        "dataset_valid",
        "dataset_message",
        "filtered_csv",
        "selected_checkpoint",
        "current_training_run",
        "last_used_checkpoint",
        "last_log_paths",
        "stage_states",
        "latest_eval_metrics",
        "examples_to_display",
        "evaluation_mode",
        "yolo_weights",
        "openclip_batch_size",
        "openclip_recompute_active",
        "training_config",
        "training_mode",
        "resume_training_config",
        "force_train_new_run",
        "new_run_config_source_checkpoint",
        "duplicate_checkbox_nonce",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = stored.get(key, value) if key in persistable_keys else value

    # Ensure newly introduced stages/keys exist when loading an older flow file.
    for stage in STAGES:
        st.session_state.stage_states.setdefault(stage, "not_started")
    for key, value in DEFAULT_TRAINING_CONFIG.items():
        st.session_state.training_config.setdefault(key, value)

    process = st.session_state.process
    if process and is_process_running(process.get("pid")):
        label = STAGE_KEY_TO_LABEL.get(process.get("stage"))
        if label:
            st.session_state.stage = label
            st.session_state.stage_states[label] = "stopping" if process.get("requested_stop") else "running"
            st.session_state.last_log_paths[process["stage"]] = process.get("log_path")


def flow_snapshot() -> dict[str, Any]:
    keys = [
        "stage",
        "dataset_root",
        "dataset_valid",
        "dataset_message",
        "filtered_csv",
        "selected_checkpoint",
        "current_training_run",
        "last_used_checkpoint",
        "last_log_paths",
        "stage_states",
        "latest_eval_metrics",
        "examples_to_display",
        "evaluation_mode",
        "yolo_weights",
        "openclip_batch_size",
        "openclip_recompute_active",
        "training_config",
        "training_mode",
        "resume_training_config",
        "force_train_new_run",
        "new_run_config_source_checkpoint",
        "duplicate_checkbox_nonce",
    ]
    return {key: st.session_state.get(key) for key in keys}


def save_flow() -> None:
    persist_flow_state(PROJECT_ROOT, flow_snapshot())


def clear_stage_log(stage_key: str) -> None:
    """Forget an old UI log when a new action starts or an artifact is reused."""
    st.session_state.last_log_paths.pop(stage_key, None)


def clear_logs_after(stage: str) -> None:
    """Clear stale downstream logs after an upstream choice invalidates them."""
    stage_to_process = {
        "YOLO": "yolo",
        "OpenCLIP": "openclip",
        "Training": "training",
        "Evaluation": "evaluation",
    }
    index = STAGES.index(stage)
    for downstream in STAGES[index + 1 :]:
        key = stage_to_process.get(downstream)
        if key:
            clear_stage_log(key)


def clear_evaluation_view() -> None:
    """Clear stale evaluation-only UI data after checkpoint or run context changes."""
    clear_stage_log("evaluation")
    st.session_state.latest_eval_metrics = {}
    set_stage_state("Evaluation", "not_started")
    set_stage_state("Results", "not_started")


def clear_training_view() -> None:
    """Clear stale training-only UI data before preparing a separate run."""
    clear_stage_log("training")
    st.session_state.current_training_run = None
    set_stage_state("Training", "not_started")
    clear_evaluation_view()


def set_stage_state(stage: str, value: str) -> None:
    st.session_state.stage_states[stage] = value


def stage_state(stage: str) -> str:
    return st.session_state.stage_states.get(stage, "not_started")


def detect_yolo_artifact_state() -> str:
    """Classify the saved YOLO CSV without confusing a partial file with a completed run."""
    if not st.session_state.dataset_valid:
        return "not_started"
    csv_path = find_filtered_csv(st.session_state.dataset_root)
    if not csv_path:
        return "not_started"
    summary = read_csv_summary(csv_path)
    total = count_trajectory_folders(st.session_state.dataset_root)
    if total > 0 and 0 < summary.rows < total:
        return "partial"
    if total > 0 and summary.rows >= total:
        return "existing"
    return "not_started"


def detect_openclip_artifact_state() -> str:
    """Classify the on-disk OpenCLIP cache as absent, partial or complete."""
    csv_path = st.session_state.filtered_csv or find_filtered_csv(st.session_state.dataset_root)
    summary = read_csv_summary(csv_path)
    if summary.success <= 0:
        return "not_started"
    cached = min(count_embeddings(st.session_state.dataset_root), summary.success)
    if cached >= summary.success:
        return "existing"
    if cached > 0:
        return "partial"
    return "not_started"


def action_spacer() -> None:
    st.markdown('<div class="action-spacer"></div>', unsafe_allow_html=True)


def artifact_date_note(label: str, value: str | None) -> str:
    """Return a compact artifact timestamp label for decision and summary screens."""
    return f"{label}: {value}" if value else f"{label}: unavailable"


def active_process_started(stage_key: str) -> str | None:
    process = st.session_state.process
    if process and process.get("stage") == stage_key:
        return process_started_text(process.get("started_at"))
    return None


def three_action_columns():
    """Give the forward action a little more width so long labels stay on one line."""
    return st.columns([1.0, 1.0, 1.0], gap="large")


def active_process() -> dict | None:
    process = st.session_state.process
    return process if process and is_process_running(process.get("pid")) else None


def active_stage_key() -> str | None:
    process = active_process()
    return process.get("stage") if process else None


def active_mode() -> str:
    process = active_process()
    return str((process or {}).get("metadata", {}).get("mode", ""))


def latest_lines(stage_key: str, max_lines: int | None = 180) -> list[str]:
    """Read the latest log lines. Evaluation can request the full log.

    The Evaluation screen intentionally shows all saved examples, so it asks
    for max_lines=None. Other stages keep a tail to stay lightweight.
    """
    process = st.session_state.process
    if process and process.get("stage") == stage_key:
        return read_log_tail(process.get("log_path"), max_lines=max_lines)
    return read_log_tail(st.session_state.last_log_paths.get(stage_key), max_lines=max_lines)


def tone_for(state: str) -> str:
    if state in {"completed", "ready"}:
        return "green"
    if state == "failed":
        return "red"
    if state in {"stopped", "existing", "partial", "stopping"}:
        return "amber"
    return "cyan"


def safe_float_values(value: Any) -> list[float]:
    """Parse metric values from None, text, numpy arrays, or Python lists.

    Evaluation examples can now arrive from JSON as real Python lists.  A list is
    not hashable, so it must be handled before any membership test such as
    ``value in {...}``.  This prevents the Evaluation page from stopping before
    the second chart and the log are rendered.
    """
    if value is None:
        return []

    if isinstance(value, str):
        if value.strip() in {"", "None", "null"}:
            return []
        cleaned = value.replace(",", " ").replace("[", " ").replace("]", " ")
        values: list[float] = []
        for token in cleaned.split():
            try:
                values.append(float(token))
            except (TypeError, ValueError):
                continue
        return values

    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                values.extend(safe_float_values(item))
                continue
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return values

    # Support numpy arrays / tensors without importing numpy or torch here.
    if hasattr(value, "tolist"):
        try:
            return safe_float_values(value.tolist())
        except Exception:
            return []

    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def apply_dataset_selection(path_value: str) -> None:
    """Normalize, validate and store a selected raw-trajectory folder."""
    normalized = normalize_dataset_root(path_value)
    valid, message = validate_dataset_root(normalized)
    st.session_state.dataset_root = normalized
    st.session_state.dataset_valid = valid
    st.session_state.dataset_message = message
    st.session_state.filtered_csv = find_filtered_csv(normalized) if valid else None
    # Invalidate the old flow first. Then expose the correct YOLO decision
    # state for the newly selected folder. The previous order reset YOLO back
    # to NOT_STARTED after detecting an existing CSV, which made the page draw
    # a misleading 100% progress bar from saved rows.
    st.session_state.stage_states = _initial_stage_states()
    reset_downstream("Dataset")
    set_stage_state("Dataset", "completed" if valid else "not_started")
    if valid:
        set_stage_state("YOLO", detect_yolo_artifact_state())
    # A finished process descriptor from a previous folder must not mutate the
    # new flow during the next render.
    if not active_process():
        st.session_state.process = None
        clear_current_process(PROJECT_ROOT)
    save_flow()


def reset_downstream(after_stage: str) -> None:
    """Invalidate later stages when a new upstream process starts."""
    index = STAGES.index(after_stage)
    for stage in STAGES[index + 1 :]:
        set_stage_state(stage, "not_started")
    clear_logs_after(after_stage)
    st.session_state.latest_eval_metrics = {}
    if after_stage in {"Dataset", "YOLO", "OpenCLIP"}:
        st.session_state.selected_checkpoint = None
        st.session_state.current_training_run = None
    if after_stage in {"Dataset", "YOLO"}:
        st.session_state.openclip_recompute_active = False


def unlock_after_yolo() -> None:
    set_stage_state("OpenCLIP", detect_openclip_artifact_state())


def unlock_after_openclip() -> None:
    checkpoints = discover_checkpoints()
    set_stage_state("Model", "existing" if checkpoints else "not_started")
    if checkpoints and st.session_state.selected_checkpoint not in checkpoints:
        preferred = st.session_state.get("last_used_checkpoint")
        st.session_state.selected_checkpoint = preferred if preferred in checkpoints else next(iter(checkpoints))


def refresh_process() -> bool:
    """Finalize a process after it exits. Return True when UI state changed."""
    process = st.session_state.process
    if not process or is_process_running(process.get("pid")):
        return False

    stage_key = process["stage"]
    ui_stage = STAGE_KEY_TO_LABEL[stage_key]
    lines = read_log_tail(process.get("log_path"), max_lines=None if stage_key == "evaluation" else 180)
    st.session_state.last_log_paths[stage_key] = process.get("log_path")

    if process.get("requested_stop"):
        set_stage_state(ui_stage, "stopped")
        if stage_key == "training":
            remember_checkpoint(st.session_state.current_training_run or st.session_state.training_config.get("run_name"))
    elif process_completed_successfully(stage_key, lines):
        set_stage_state(ui_stage, "completed")
        if stage_key == "yolo":
            st.session_state.filtered_csv = find_filtered_csv(st.session_state.dataset_root)
            unlock_after_yolo()
        elif stage_key == "openclip":
            st.session_state.openclip_recompute_active = False
            unlock_after_openclip()
        elif stage_key == "training":
            run_name = st.session_state.current_training_run or st.session_state.training_config.get("run_name")
            remember_checkpoint(run_name)
            set_stage_state("Evaluation", "not_started")
        elif stage_key == "evaluation":
            st.session_state.latest_eval_metrics = parse_evaluation_metrics(lines)
            set_stage_state("Results", "completed")
    else:
        set_stage_state(ui_stage, "failed")

    st.session_state.process = None
    clear_current_process(PROJECT_ROOT)
    save_flow()
    return True


def launch(stage_key: str, command: list[str], ui_stage: str, *, mode: str) -> None:
    if active_process():
        st.error("Another process is already running. Stop it safely before starting a new action.")
        return
    clear_stage_log(stage_key)
    process = start_process(command, PROJECT_ROOT, stage_key, metadata={"mode": mode})
    st.session_state.process = process
    st.session_state.last_log_paths[stage_key] = process.get("log_path")
    st.session_state.stage = ui_stage
    set_stage_state(ui_stage, "running")
    save_flow()
    st.rerun()


def request_stop() -> None:
    process = st.session_state.process
    if not process:
        return
    if stop_process(process.get("pid")):
        process["requested_stop"] = "stopped"
        st.session_state.process = process
        label = STAGE_KEY_TO_LABEL.get(process.get("stage"))
        if label:
            set_stage_state(label, "stopping")
        persist_process(PROJECT_ROOT, process)
        save_flow()
    # Do not force a full-page rerun here. The live fragment rerenders
    # naturally, avoiding a visible jump while the process changes to STOPPING.


def checkpoint_info() -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    checkpoints = discover_checkpoints()
    selected = checkpoints.get(st.session_state.selected_checkpoint or "")
    return checkpoints, selected


def checkpoint_ready() -> bool:
    _, selected = checkpoint_info()
    return bool(selected and selected.get("has_model"))


def remember_checkpoint(run_name: str | None) -> None:
    """Remember the last checkpoint that was selected or used by a real action.

    Never mutate the selectbox widget key here. Streamlit forbids changing a
    widget-backed session-state value after that widget has already been
    instantiated during the current render. The Model page performs the safe
    synchronization before it creates the selectbox on the next render.
    """
    if not run_name:
        return
    st.session_state.last_used_checkpoint = run_name
    st.session_state.selected_checkpoint = run_name
    st.session_state.checkpoint_selector_sync_pending = True


def can_open_stage(stage: str) -> bool:
    if active_process():
        return stage == st.session_state.stage
    if stage == "Dataset":
        return True
    if stage == "YOLO":
        return bool(st.session_state.dataset_valid)
    if stage == "OpenCLIP":
        return stage_state("YOLO") in {"completed", "ready"}
    if stage == "Model":
        return stage_state("OpenCLIP") in {"completed", "ready"}
    if stage == "Training":
        return stage_state("Model") == "completed"
    if stage == "Evaluation":
        return stage_state("Model") == "completed" and checkpoint_ready()
    if stage == "Results":
        return stage_state("Evaluation") == "completed" and bool(st.session_state.latest_eval_metrics)
    return False


def reset_pipeline_view() -> None:
    """Reset only the UI flow. Keep CSV files, embeddings and checkpoints on disk."""
    if active_process():
        st.warning("Stop the active process safely before resetting the UI view.")
        return
    st.session_state.stage = "Dataset"
    st.session_state.dataset_root = ""
    st.session_state.dataset_valid = False
    st.session_state.dataset_message = "Choose a raw trajectory folder first."
    st.session_state.filtered_csv = None
    st.session_state.selected_checkpoint = None
    st.session_state.current_training_run = None
    st.session_state.last_used_checkpoint = None
    st.session_state.last_log_paths = {}
    st.session_state.stage_states = _initial_stage_states()
    st.session_state.latest_eval_metrics = {}
    st.session_state.openclip_recompute_active = False
    st.session_state.training_mode = "new"
    st.session_state.resume_training_config = {}
    st.session_state.force_train_new_run = False
    st.session_state.new_run_config_source_checkpoint = None
    st.session_state.duplicate_checkbox_nonce = 0
    st.session_state.folder_picker_error = None
    clear_current_process(PROJECT_ROOT)
    clear_flow_state(PROJECT_ROOT)
    save_flow()


def page_shell() -> None:
    """Compact header: brand and pipeline stepper share one row."""
    left, right = st.columns([1.05, 1.95], gap="large")
    with left:
        brand(st.session_state.stage)
    with right:
        stage_stepper(st.session_state.stage, st.session_state.stage_states)


def dataset_page() -> None:
    stage_header("Start from a dataset folder", "Choose a BridgeData raw-trajectory folder before running preprocessing.", "DATASET", "cyan")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        section_card("Dataset / Run folder", "Choose one raw-trajectory folder. RoboMamba detects only its own final_mamba_dataset.csv inside that folder.")

        if st.button("Browse folder", type="primary", use_container_width=True, key="browse_dataset_folder"):
            selected, error = browse_server_folder(st.session_state.dataset_root or DEFAULT_DATASET_ROOT)
            st.session_state.folder_picker_error = error
            if selected:
                apply_dataset_selection(selected)
            else:
                save_flow()
            st.rerun()

        if st.session_state.folder_picker_error:
            st.warning(st.session_state.folder_picker_error)
            with st.expander("Manual path fallback", expanded=True):
                manual = st.text_input(
                    "Server dataset folder",
                    value=st.session_state.dataset_root,
                    placeholder=DEFAULT_DATASET_ROOT,
                    help="Use this fallback only when the native folder picker cannot open on the server desktop.",
                )
                if st.button("Validate manual path", use_container_width=True, disabled=not bool(manual.strip())):
                    apply_dataset_selection(manual.strip())
                    st.rerun()

        section_card("Selected folder")
        path_box(st.session_state.dataset_root or "No folder selected")

        if st.session_state.dataset_valid:
            st.success(st.session_state.dataset_message)
            csv_path = st.session_state.filtered_csv
            if csv_path:
                summary = read_csv_summary(csv_path)
                st.info("A compatible final_mamba_dataset.csv was detected automatically inside the selected folder. " + artifact_date_note("Last updated", file_modified_text(csv_path)) + ". You can reuse it, resume filtering, or rebuild it from YOLO.")
                metrics([
                    ("CSV rows", summary.rows, "saved"),
                    ("SUCCESS", summary.success, "usable"),
                    ("FAILURE", summary.failure, "filtered"),
                    ("SKIPPED", summary.skipped, "not usable"),
                ])
            else:
                st.info("No compatible filtered CSV was found inside the selected folder. Continue to YOLO to create one.")
            if st.button("Continue to YOLO", type="primary", use_container_width=True):
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
        else:
            st.warning(st.session_state.dataset_message)
            st.button("Choose a folder first", disabled=True, use_container_width=True)
    with right:
        robot_panel("Dataset context", "Folder validation before preprocessing")


def yolo_page() -> None:
    state = stage_state("YOLO")
    stage_header("Dataset scan and YOLO validation", "Filter trajectories with the fixed project YOLO model and validate gripper or motor signals.", state.upper(), tone_for(state))
    if not st.session_state.dataset_valid:
        st.warning("Select and validate a dataset folder first.")
        if st.button("Back to dataset", use_container_width=True):
            st.session_state.stage = "Dataset"
            save_flow()
            st.rerun()
        return

    left, right = st.columns([1.05, 1], gap="large")
    # Render the visual panel before the interactive content. Some decision
    # states return early after drawing their actions; rendering the right
    # column first keeps the robot visible in those states as well.
    with right:
        robot_panel("Filtering context", "Valid trajectories continue to OpenCLIP")
    with left:
        csv_path = find_filtered_csv(st.session_state.dataset_root)
        summary = read_csv_summary(csv_path)
        trajectory_count = count_trajectory_folders(st.session_state.dataset_root)
        csv_updated = file_modified_text(csv_path)
        lines = latest_lines("yolo")
        parsed = parse_yolo_progress(lines)
        mode = active_mode()

        # Defensive safeguard for older saved UI-state files: whenever a
        # compatible CSV exists for a freshly selected folder and no YOLO
        # process is active, present the explicit reuse/rebuild decision screen
        # instead of deriving a completed-looking progress bar from CSV rows.
        if state == "not_started" and csv_path and not active_process():
            state = detect_yolo_artifact_state()
            set_stage_state("YOLO", state)
            save_flow()

        section_card("Selected dataset folder")
        path_box(st.session_state.dataset_root)

        # A previously generated CSV is a decision point, not a newly finished
        # scan.  Showing a full progress bar immediately after entering the page
        # looked like YOLO had just run.  Keep the choices explicit instead.
        if state == "existing":
            section_card("Existing filtered CSV detected", artifact_date_note("CSV last updated", csv_updated))
            st.info("Choose whether to reuse the existing CSV or rebuild it with the fixed YOLO model. A new scan must finish before OpenCLIP can be opened.")
            metrics([
                ("CSV rows", summary.rows, "saved"),
                ("Approved", summary.success, "usable"),
                ("Rejected", summary.failure, "filtered"),
                ("Skipped", summary.skipped, "not usable"),
            ])
            with st.expander("YOLO configuration"):
                st.caption("The YOLO weights are fixed by the project and cannot be changed from the UI.")
                path_box(str(DEFAULT_YOLO_WEIGHTS.relative_to(PROJECT_ROOT)), spacious=True)
                if not DEFAULT_YOLO_WEIGHTS.exists():
                    st.error("The fixed YOLO weights file was not found. Add best.pt before starting a scan.")
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to dataset", key="back_yolo_existing", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Rebuild with YOLO", key="restart_yolo_existing", use_container_width=True, disabled=not DEFAULT_YOLO_WEIGHTS.exists()):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, str(DEFAULT_YOLO_WEIGHTS), restart=True), "YOLO", mode="restart")
            if c3.button("Use existing CSV", key="reuse_yolo_existing", type="primary", use_container_width=True):
                clear_stage_log("yolo")
                set_stage_state("YOLO", "ready")
                st.session_state.filtered_csv = csv_path
                unlock_after_yolo()
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
            return

        if state == "partial":
            done = min(summary.rows, trajectory_count)
            total = trajectory_count
            ratio = done / total if total else 0.0
            section_card("Partial YOLO scan detected", artifact_date_note("Partial CSV last saved", csv_updated))
            st.warning(f"{done:,} of {total:,} trajectories were already processed. Resume from the saved CSV or restart the scan from the beginning.")
            progress_card(ratio, "Saved YOLO progress", f"{done:,} / {total:,} trajectories")
            metrics([
                ("Saved rows", summary.rows, "partial CSV"),
                ("Approved", summary.success, "usable"),
                ("Rejected", summary.failure, "filtered"),
                ("Skipped", summary.skipped, "not usable"),
            ])
            log_box(lines, "Partial CSV detected. Resume to continue filtering.", title="Saved scan log")
            with st.expander("YOLO configuration"):
                st.caption("The YOLO weights are fixed by the project and cannot be changed from the UI.")
                path_box(str(DEFAULT_YOLO_WEIGHTS.relative_to(PROJECT_ROOT)), spacious=True)
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to dataset", key="back_yolo_partial", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Restart from beginning", key="restart_yolo_partial", use_container_width=True, disabled=not DEFAULT_YOLO_WEIGHTS.exists()):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, str(DEFAULT_YOLO_WEIGHTS), restart=True), "YOLO", mode="restart")
            if c3.button("Resume scan", key="resume_yolo_partial", type="primary", use_container_width=True, disabled=not DEFAULT_YOLO_WEIGHTS.exists()):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, str(DEFAULT_YOLO_WEIGHTS)), "YOLO", mode="resume")
            return

        if state == "ready":
            section_card("YOLO output ready", artifact_date_note("CSV last updated", csv_updated))
            st.success("Filtered CSV is ready for OpenCLIP. Rebuild only when you intentionally want to run YOLO again.")
            metrics([
                ("CSV rows", summary.rows, "saved"),
                ("Approved", summary.success, "usable"),
                ("Rejected", summary.failure, "filtered"),
                ("Skipped", summary.skipped, "not usable"),
            ])
            with st.expander("YOLO configuration"):
                st.caption("The YOLO weights are fixed by the project and cannot be changed from the UI.")
                path_box(str(DEFAULT_YOLO_WEIGHTS.relative_to(PROJECT_ROOT)), spacious=True)
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to dataset", key="back_yolo_ready", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Rebuild with YOLO", key="restart_yolo_ready", use_container_width=True, disabled=not DEFAULT_YOLO_WEIGHTS.exists()):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, str(DEFAULT_YOLO_WEIGHTS), restart=True), "YOLO", mode="restart")
            if c3.button("Continue to OpenCLIP", key="continue_yolo_ready", type="primary", use_container_width=True):
                st.session_state.filtered_csv = csv_path
                unlock_after_yolo()
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
            return

        if state == "not_started":
            done, total, ratio = 0, trajectory_count, 0.0
        elif state == "completed":
            done, total, ratio = trajectory_count, trajectory_count, 1.0
        elif state in {"running", "stopping"} and mode == "restart":
            done, total = parsed["done"], parsed["total"] or trajectory_count
            ratio = done / total if total else 0.0
        else:
            done = parsed["done"] or summary.rows
            total = parsed["total"] or trajectory_count
            ratio = done / total if total else 0.0

        title = "YOLO filter progress" if state != "existing" else "Existing filtered CSV detected"
        progress_card(ratio, title, f"{done:,} / {total:,} trajectories")
        if state in {"running", "stopping"}:
            started = active_process_started("yolo")
            if started:
                st.caption(f"Current YOLO run started: {started}")
        elif csv_updated:
            st.caption(f"CSV last saved: {csv_updated}")
        metrics([
            ("Scanned", summary.rows, "saved rows"),
            ("Approved", summary.success, "usable"),
            ("Rejected", summary.failure, "filtered"),
            ("Skipped", summary.skipped, "not usable"),
        ])
        log_box(lines, "Waiting to start YOLO scan...", title="Scan log")
        with st.expander("YOLO configuration"):
            st.caption("The YOLO weights are fixed by the project and cannot be changed from the UI.")
            path_box(str(DEFAULT_YOLO_WEIGHTS.relative_to(PROJECT_ROOT)), spacious=True)
            if not DEFAULT_YOLO_WEIGHTS.exists():
                st.error("The fixed YOLO weights file was not found. Add best.pt before starting the scan.")

        weights_path = str(DEFAULT_YOLO_WEIGHTS)
        weights_missing = not DEFAULT_YOLO_WEIGHTS.exists()

        if state == "running":
            st.button("Stop safely", on_click=request_stop, key="stop_yolo", use_container_width=True)
        elif state == "stopping":
            st.button("Stopping safely...", disabled=True, use_container_width=True)
        elif state == "completed":
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to dataset", key="back_yolo_completed", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Restart scan", key="restart_yolo_completed", use_container_width=True, disabled=weights_missing):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, weights_path, restart=True), "YOLO", mode="restart")
            if c3.button("Continue to OpenCLIP", key="continue_yolo", type="primary", use_container_width=True):
                st.session_state.filtered_csv = csv_path
                unlock_after_yolo()
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
        elif state == "stopped":
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to dataset", key="back_yolo", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Restart", key="restart_yolo", use_container_width=True, disabled=weights_missing):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, weights_path, restart=True), "YOLO", mode="restart")
            if c3.button("Resume", key="resume_yolo", type="primary", use_container_width=True, disabled=weights_missing):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, weights_path), "YOLO", mode="resume")
        elif state == "failed":
            if st.button("Back to dataset", key="failed_yolo_back", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
        else:
            c1, c2 = st.columns(2, gap="medium")
            if c1.button("Back to dataset", key="back_yolo_before_start", use_container_width=True):
                st.session_state.stage = "Dataset"
                save_flow()
                st.rerun()
            if c2.button("Start scan", key="start_yolo", type="primary", use_container_width=True, disabled=weights_missing):
                reset_downstream("YOLO")
                launch("yolo", build_yolo_command(st.session_state.dataset_root, weights_path), "YOLO", mode="resume")


def start_openclip_recompute(csv_path: str, batch_size: int) -> None:
    """Start a fresh cache rebuild that can later resume from partial files.

    We intentionally delete only RoboMamba's OpenCLIP cache files before the
    rebuild. The standard extractor then runs without --overwrite. If the user
    stops it, a later Resume skips regenerated trajectories and continues from
    the remaining ones instead of completing immediately from an old cache.
    """
    clear_embedding_cache(st.session_state.dataset_root)
    st.session_state.openclip_recompute_active = True
    reset_downstream("OpenCLIP")
    launch("openclip", build_openclip_command(csv_path, batch_size), "OpenCLIP", mode="recompute")


def openclip_page() -> None:
    state = stage_state("OpenCLIP")
    stage_header("Generate OpenCLIP features", "Reuse cached embeddings or extract a new feature cache while preserving frame alignment.", state.upper(), tone_for(state))
    csv_path = st.session_state.filtered_csv or find_filtered_csv(st.session_state.dataset_root)
    if not csv_path or stage_state("YOLO") not in {"completed", "ready"}:
        st.warning("Complete or explicitly reuse the YOLO output first.")
        if st.button("Back to YOLO", key="openclip_missing_yolo_back", use_container_width=True):
            st.session_state.stage = "YOLO"
            save_flow()
            st.rerun()
        return

    left, right = st.columns([1.05, 1], gap="large")
    # Keep the robot visible on the existing-cache decision screen too.
    # Previously the early return inside the left column prevented the right
    # column from being rendered.
    with right:
        robot_panel("Detected artifacts", "CSV + images + robot state + embeddings")
    with left:
        summary = read_csv_summary(csv_path)
        cached = count_embeddings(st.session_state.dataset_root)
        cache_updated = latest_embedding_modified_text(st.session_state.dataset_root)
        lines = latest_lines("openclip")
        parsed = parse_embedding_progress(lines)
        mode = active_mode()
        fixed_batch_size = 32
        st.session_state.openclip_batch_size = fixed_batch_size

        if state == "not_started" and not active_process():
            detected = detect_openclip_artifact_state()
            if detected != "not_started":
                set_stage_state("OpenCLIP", detected)
                state = detected
                save_flow()

        # Existing cache is a decision point, not a completed process view.
        # Do not show a misleading 100% progress bar before the user chooses
        # whether to reuse the cache or recompute it.
        if state == "existing":
            section_card("Existing OpenCLIP cache detected", artifact_date_note("Cache last updated", cache_updated))
            st.info(f"{min(cached, summary.success):,} cached embeddings were detected for {summary.success:,} SUCCESS trajectories. Choose whether to reuse them or recompute the cache.")
            metrics([
                ("SUCCESS trajectories", summary.success, "input"),
                ("Cached embeddings", min(cached, summary.success), "available on disk"),
            ])
            with st.expander("OpenCLIP configuration"):
                st.caption("Batch size is fixed to 32 for this project UI. Change it only from the command line when running a deliberate experiment.")
                path_box("OpenCLIP batch size: 32", spacious=True)
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to YOLO", key="back_openclip_existing", use_container_width=True):
                set_stage_state("YOLO", "ready")
                clear_stage_log("yolo")
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Recompute all", key="recompute_openclip_existing", use_container_width=True):
                start_openclip_recompute(csv_path, fixed_batch_size)
            if c3.button("Use existing cache", key="reuse_openclip", type="primary", use_container_width=True):
                clear_stage_log("openclip")
                st.session_state.openclip_recompute_active = False
                set_stage_state("OpenCLIP", "ready")
                unlock_after_openclip()
                st.session_state.stage = "Model"
                save_flow()
                st.rerun()
            return

        if state == "partial":
            done = min(cached, summary.success)
            total = summary.success
            ratio = done / total if total else 0.0
            section_card("Partial OpenCLIP cache detected", artifact_date_note("Partial cache last saved", cache_updated))
            st.warning(f"{done:,} of {total:,} embeddings are available. Resume extraction or recompute the entire cache.")
            progress_card(ratio, "Saved OpenCLIP progress", f"{done:,} / {total:,} trajectories embedded")
            metrics([
                ("SUCCESS trajectories", total, "input"),
                ("Saved embeddings", done, "partial cache"),
                ("Remaining", max(0, total - done), "pending"),
            ])
            log_box(lines, "Partial cache detected. Resume extraction to continue.", title="Saved extraction log")
            with st.expander("OpenCLIP configuration"):
                st.caption("Batch size is fixed to 32 for this project UI.")
                path_box("OpenCLIP batch size: 32", spacious=True)
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to YOLO", key="back_openclip_partial", use_container_width=True):
                set_stage_state("YOLO", "ready")
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Recompute all", key="recompute_openclip_partial_explicit", use_container_width=True):
                start_openclip_recompute(csv_path, fixed_batch_size)
            if c3.button("Resume extraction", key="resume_openclip_partial_explicit", type="primary", use_container_width=True):
                reset_downstream("OpenCLIP")
                launch("openclip", build_openclip_command(csv_path, fixed_batch_size), "OpenCLIP", mode="resume")
            return

        if state == "ready":
            section_card("OpenCLIP feature cache ready", artifact_date_note("Cache last updated", cache_updated))
            st.success("Embeddings are ready for model configuration. Recompute only when you intentionally want to rebuild the cache.")
            metrics([
                ("SUCCESS trajectories", summary.success, "input"),
                ("Cached embeddings", min(cached, summary.success), "ready on disk"),
            ])
            with st.expander("OpenCLIP configuration"):
                st.caption("Batch size is fixed to 32 for this project UI.")
                path_box("OpenCLIP batch size: 32", spacious=True)
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to YOLO", key="back_openclip_ready", use_container_width=True):
                set_stage_state("YOLO", "ready")
                clear_stage_log("yolo")
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Recompute all", key="recompute_openclip_ready", use_container_width=True):
                start_openclip_recompute(csv_path, fixed_batch_size)
            if c3.button("Continue to model", key="continue_openclip_ready", type="primary", use_container_width=True):
                unlock_after_openclip()
                st.session_state.stage = "Model"
                save_flow()
                st.rerun()
            return

        if state == "not_started":
            done, total, ratio = 0, summary.success, 0.0
        elif state == "completed":
            done, total, ratio = summary.success, summary.success, 1.0
        elif state in {"running", "stopping"}:
            done = parsed["done"]
            total = parsed["total"] or summary.success
            ratio = parsed["ratio"] if parsed["total"] else (done / total if total else 0.0)
        elif state in {"stopped", "failed", "partial"}:
            done = parsed["done"] if parsed["done"] else min(cached, summary.success)
            total = parsed["total"] or summary.success
            ratio = done / total if total else 0.0
        else:
            done, total, ratio = 0, summary.success, 0.0

        remaining = max(0, total - done)
        title = "Feature cache status" if state != "completed" else "Feature extraction completed"
        progress_card(ratio, title, f"{done:,} / {total:,} trajectories embedded")
        if state in {"running", "stopping"}:
            started = active_process_started("openclip")
            if started:
                st.caption(f"Current OpenCLIP run started: {started}")
        elif cache_updated:
            st.caption(f"Cache last saved: {cache_updated}")

        if state in {"running", "stopping"}:
            # Keep the live view intentionally small and unambiguous. Existing
            # files on disk can include the previous cache during overwrite.
            metrics([
                ("SUCCESS trajectories", summary.success, "input"),
                ("Current run progress", done, "current action"),
                ("Remaining", remaining, "current action"),
            ])
            if mode == "overwrite":
                st.caption("Recomputing the cache. Existing files may remain on disk until they are replaced; the progress bar shows only the current run.")
        elif state == "completed":
            metrics([
                ("SUCCESS trajectories", summary.success, "input"),
                ("Cached embeddings", min(cached, summary.success), "ready"),
                ("Remaining", 0, "complete"),
            ])
        else:
            metrics([
                ("SUCCESS trajectories", summary.success, "input"),
                ("Available embeddings", min(cached, summary.success), "saved on disk"),
                ("Remaining", remaining, "current action"),
            ])

        log_box(lines, "Waiting to generate embeddings...", title="Scan log")
        with st.expander("OpenCLIP configuration"):
            st.caption("Batch size is fixed to 32 for this project UI. Change it only from the command line when running a deliberate experiment.")
            path_box("OpenCLIP batch size: 32", spacious=True)

        if state == "running":
            st.button("Stop safely", on_click=request_stop, key="stop_openclip", use_container_width=True)
        elif state == "stopping":
            st.button("Stopping safely...", disabled=True, use_container_width=True)
        elif state == "completed":
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to YOLO", key="back_openclip_completed", use_container_width=True):
                set_stage_state("YOLO", "ready")
                clear_stage_log("yolo")
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Recompute all", key="recompute_openclip_completed", use_container_width=True):
                start_openclip_recompute(csv_path, fixed_batch_size)
            if c3.button("Continue to model", key="continue_openclip", type="primary", use_container_width=True):
                unlock_after_openclip()
                st.session_state.stage = "Model"
                save_flow()
                st.rerun()
        elif state == "stopped":
            action_spacer()
            c1, c2, c3 = three_action_columns()
            if c1.button("Back to YOLO", key="back_openclip", use_container_width=True):
                set_stage_state("YOLO", "ready")
                clear_stage_log("yolo")
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Recompute", key="recompute_openclip", use_container_width=True):
                start_openclip_recompute(csv_path, fixed_batch_size)
            if c3.button("Resume", key="resume_openclip", type="primary", use_container_width=True):
                reset_downstream("OpenCLIP")
                launch("openclip", build_openclip_command(csv_path, fixed_batch_size), "OpenCLIP", mode="resume")
        elif state == "failed":
            if st.button("Back to YOLO", key="failed_openclip_back", use_container_width=True):
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
        else:
            c1, c2 = st.columns(2, gap="medium")
            if c1.button("Back to YOLO", key="back_openclip_before_start", use_container_width=True):
                st.session_state.stage = "YOLO"
                save_flow()
                st.rerun()
            if c2.button("Generate embeddings", key="start_openclip", type="primary", use_container_width=True):
                reset_downstream("OpenCLIP")
                launch("openclip", build_openclip_command(csv_path, fixed_batch_size), "OpenCLIP", mode="resume")



NEW_RUN_WIDGET_KEYS = {
    "seq_length": "new_run_seq_length",
    "d_model": "new_run_d_model",
    "batch_size": "new_run_batch_size",
    "learning_rate": "new_run_learning_rate",
    "max_epochs": "new_run_max_epochs",
    "patience": "new_run_patience",
}

DUPLICATE_NEW_RUN_KEY = "train_duplicate_as_new_run"

def _set_new_run_widget_values(config: dict[str, Any]) -> None:
    """Synchronize editable new-run widgets before they are instantiated."""
    defaults = dict(DEFAULT_TRAINING_CONFIG)
    merged = {**defaults, **(config or {})}
    int_fields = ("seq_length", "d_model", "batch_size", "max_epochs", "patience")
    for field in int_fields:
        try:
            value = int(merged.get(field, defaults.get(field)))
        except (TypeError, ValueError):
            value = int(defaults.get(field))
        st.session_state[NEW_RUN_WIDGET_KEYS[field]] = value
    try:
        lr = float(merged.get("learning_rate", defaults.get("learning_rate", 0.001)))
    except (TypeError, ValueError):
        lr = 0.001
    st.session_state[NEW_RUN_WIDGET_KEYS["learning_rate"]] = lr

def _read_new_run_widget_values() -> dict[str, Any]:
    """Read the live editable new-run values from Streamlit widgets."""
    config = dict(DEFAULT_TRAINING_CONFIG)
    for field, key in NEW_RUN_WIDGET_KEYS.items():
        if key in st.session_state:
            config[field] = st.session_state[key]
    config["seq_length"] = int(config.get("seq_length", 10))
    config["d_model"] = int(config.get("d_model", 128))
    config["batch_size"] = int(config.get("batch_size", 16))
    config["learning_rate"] = float(config.get("learning_rate", 0.001))
    config["max_epochs"] = int(config.get("max_epochs", 20))
    config["patience"] = int(config.get("patience", 5))
    config["run_name"] = None
    return config

def _sync_training_config_from_checkpoint(saved_config: dict[str, Any]) -> None:
    """Use a checkpoint configuration as the starting point for a new run."""
    editable = {"seq_length", "d_model", "batch_size", "learning_rate", "max_epochs", "patience"}
    for key in editable:
        if key in saved_config and saved_config[key] is not None:
            st.session_state.training_config[key] = saved_config[key]
    st.session_state.training_config["run_name"] = None
    _set_new_run_widget_values(st.session_state.training_config)


def _ensure_new_run_config_source(selected: str | None, saved_config: dict[str, Any]) -> None:
    """Keep the editable new-run panel aligned with the selected checkpoint.

    The new-run panel is only a starting point for a separate future run.
    When the user first opens Model, or manually selects another checkpoint,
    the editable fields should start from the selected checkpoint values.
    This avoids showing unrelated stale values from a previous checkpoint.

    We deliberately do not mutate Streamlit widget keys here; only our own
    training_config state is updated before the form widgets are built.
    """
    if not selected or not saved_config:
        return
    if st.session_state.get("new_run_config_source_checkpoint") != selected:
        _sync_training_config_from_checkpoint(saved_config)
        st.session_state.new_run_config_source_checkpoint = selected
        st.session_state.force_train_new_run = False
        st.session_state.new_run_block_reason = ""


def _saved_config_items(config: dict[str, Any]) -> list[tuple[str, object]]:
    return [
        ("seq_length", config.get("seq_length", "--")),
        ("d_model", config.get("d_model", "--")),
        ("batch_size", config.get("batch_size", "--")),
        ("learning_rate", config.get("learning_rate", "--")),
        ("max_epochs", config.get("max_epochs", "--")),
        ("patience", config.get("patience", "--")),
    ]



def _visible_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the user-editable fields that define a visible new run."""
    keys = ("seq_length", "d_model", "batch_size", "learning_rate", "max_epochs", "patience")
    return {key: config.get(key) for key in keys}


def _duplicate_checkbox_key(run_name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(run_name))
    return f"force_train_duplicate_{safe}"


def _duplicate_checkbox_widget_key() -> str:
    return f"{DUPLICATE_NEW_RUN_KEY}_{int(st.session_state.get('duplicate_checkbox_nonce', 0))}"


def _clear_duplicate_checkbox_states() -> None:
    """Reset duplicate-training approval and rotate the checkbox widget key."""
    for key in list(st.session_state.keys()):
        if str(key).startswith(DUPLICATE_NEW_RUN_KEY) or str(key).startswith("force_train_duplicate_") or key == "force_train_duplicate_checkbox":
            try:
                del st.session_state[key]
            except Exception:
                pass
    st.session_state.duplicate_checkbox_nonce = int(st.session_state.get("duplicate_checkbox_nonce", 0)) + 1
    st.session_state.force_train_new_run = False


def _new_training_config_is_dirty(saved_config: dict[str, Any], editable_config: dict[str, Any]) -> bool:
    """Tell whether the editable new-run panel differs from the selected checkpoint."""
    if not saved_config:
        return True
    return _visible_training_config(saved_config) != _visible_training_config(editable_config)


def _checkpoint_status_label(checkpoint: dict[str, Any] | None) -> str:
    if not checkpoint:
        return "No checkpoint selected"
    if checkpoint.get("can_resume"):
        return "Interrupted training - resume available"
    if checkpoint.get("training_completed") and checkpoint.get("has_model"):
        return "Completed trained weights"
    if checkpoint.get("has_model"):
        return "Weights available"
    return "Missing trained weights"

def model_page() -> None:
    state = stage_state("Model")
    stage_header(
        "Choose model or start a new run",
        "Select trained weights for evaluation, resume an interrupted run, or create a separate experiment without modifying existing checkpoints.",
        state.upper(),
        tone_for(state),
    )
    if stage_state("OpenCLIP") not in {"completed", "ready"}:
        st.warning("Complete or explicitly reuse the OpenCLIP cache first.")
        return

    checkpoints = discover_checkpoints()
    names = list(checkpoints)
    selected_info = None
    saved_config: dict[str, Any] = {}

    if names:
        preferred = st.session_state.selected_checkpoint if st.session_state.selected_checkpoint in names else st.session_state.get("last_used_checkpoint")
        default_value = preferred if preferred in names else names[-1]
        # Synchronize the widget only before it is instantiated. This keeps the
        # last used checkpoint selected when returning from Training or
        # Evaluation without triggering StreamlitAPIException.
        if st.session_state.get("checkpoint_selector_sync_pending") or st.session_state.get("checkpoint_selector") not in names:
            st.session_state.checkpoint_selector = default_value
            st.session_state.checkpoint_selector_sync_pending = False
        selected = st.selectbox("Registered checkpoint", names, key="checkpoint_selector")
        if selected != st.session_state.selected_checkpoint:
            remember_checkpoint(selected)
            clear_evaluation_view()
            save_flow()
        selected_info = checkpoints.get(selected)
        saved_config = load_json(selected_info["config_path"]) if selected_info else {}
        _ensure_new_run_config_source(selected, saved_config)

        source_config_block(
            "Selected checkpoint",
            "Read-only saved-weight information. Existing checkpoints are never modified by the editable new-run panel below.",
            [
                ("Checkpoint", selected),
                ("Status", _checkpoint_status_label(selected_info)),
                ("Last updated", selected_info.get("last_updated") or "unavailable"),
                ("Configuration source", f"checkpoints/{selected}/config.json"),
            ],
            _saved_config_items(saved_config),
        )

        st.markdown('<div class="source-grid-title">Available actions</div>', unsafe_allow_html=True)
        if selected_info.get("can_resume"):
            st.info("This checkpoint stopped before training completed. Resume uses its locked saved configuration. To experiment with different values, open the separate new-run panel instead.")
            c1, c2 = st.columns(2, gap="large")
            if c1.button("Back to OpenCLIP", key="back_model_interrupted", use_container_width=True):
                set_stage_state("OpenCLIP", "ready")
                clear_stage_log("openclip")
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
            if c2.button("Resume interrupted training", key="resume_interrupted_training", type="primary", use_container_width=True):
                remember_checkpoint(selected)
                resume_config = load_json(selected_info["config_path"])
                st.session_state.resume_training_config = dict(resume_config)
                clear_stage_log("training")
                clear_evaluation_view()
                st.session_state.training_mode = "resume"
                st.session_state.current_training_run = selected_info["run_name"]
                set_stage_state("Model", "completed")
                set_stage_state("Training", "stopped")
                st.session_state.stage = "Training"
                save_flow()
                st.rerun()
        elif selected_info.get("has_model"):
            c1, c2 = st.columns(2, gap="large")
            if c1.button("Back to OpenCLIP", key="back_model_completed", use_container_width=True):
                set_stage_state("OpenCLIP", "ready")
                clear_stage_log("openclip")
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
            if c2.button("Evaluate selected checkpoint", key="evaluate_checkpoint_completed", type="primary", use_container_width=True):
                remember_checkpoint(selected)
                clear_evaluation_view()
                set_stage_state("Model", "completed")
                set_stage_state("Training", "skipped")
                st.session_state.stage = "Evaluation"
                save_flow()
                st.rerun()
        else:
            st.warning("This run does not contain final trained weights. Start a separate new run below or choose another checkpoint.")
            if st.button("Back to OpenCLIP", key="back_model_missing", use_container_width=True):
                set_stage_state("OpenCLIP", "ready")
                clear_stage_log("openclip")
                st.session_state.stage = "OpenCLIP"
                save_flow()
                st.rerun()
    else:
        st.info("No registered checkpoint was discovered. Configure a separate new run below.")
        if st.button("Back to OpenCLIP", key="back_model_no_checkpoint", use_container_width=True):
            set_stage_state("OpenCLIP", "ready")
            clear_stage_log("openclip")
            st.session_state.stage = "OpenCLIP"
            save_flow()
            st.rerun()

    # The new-run editor is intentionally live (not inside st.form).
    # This makes duplicate detection and Copy selected checkpoint settings behave
    # immediately, while the actual training still starts only later in Training.
    config = dict(st.session_state.training_config)
    # Initialize widget values without using a bare expression.
    # Streamlit can render bare expressions such as `... else None` as a visible "None" badge.
    if not any(key in st.session_state for key in NEW_RUN_WIDGET_KEYS.values()):
        _set_new_run_widget_values(config)

    with st.expander("Start a separate training run", expanded=not bool(names)):
        st.caption("These values create a separate checkpoint with a new generated name. Existing checkpoints are never modified or overwritten.")
        st.caption("Use this panel only when you want to train a new model. Evaluation and Resume use the selected checkpoint above, not these editable values.")
        if selected_info and saved_config:
            if st.button("Copy selected checkpoint settings", key="copy_checkpoint_settings", use_container_width=True):
                _sync_training_config_from_checkpoint(saved_config)
                st.session_state.new_run_config_source_checkpoint = selected
                _clear_duplicate_checkbox_states()
                st.session_state.new_run_block_reason = ""
                save_flow()
                st.rerun()

        c1, c2, c3 = st.columns(3, gap="medium")
        c1.number_input("Sequence length", min_value=2, max_value=100, step=1, key=NEW_RUN_WIDGET_KEYS["seq_length"], help="Number of consecutive frames supplied to Mamba.")

        d_model_options = [64, 128, 256, 512]
        if st.session_state.get(NEW_RUN_WIDGET_KEYS["d_model"]) not in d_model_options:
            st.session_state[NEW_RUN_WIDGET_KEYS["d_model"]] = 128
        c2.selectbox("Model hidden dimension (d_model)", d_model_options, key=NEW_RUN_WIDGET_KEYS["d_model"], help="Width of Mamba's hidden representation. Larger values can model richer patterns but use more GPU memory. 512 is experimental.")

        batch_options = [1, 2, 4, 8, 16, 32, 64]
        if st.session_state.get(NEW_RUN_WIDGET_KEYS["batch_size"]) not in batch_options:
            st.session_state[NEW_RUN_WIDGET_KEYS["batch_size"]] = 16
        c3.selectbox("Batch size", batch_options, key=NEW_RUN_WIDGET_KEYS["batch_size"], help="Number of training windows processed together in one optimization step. Larger values use more GPU memory.")

        c4, c5, c6 = st.columns(3, gap="medium")
        c4.number_input("Learning rate", min_value=0.000001, max_value=0.1, step=0.000001, format="%.6f", key=NEW_RUN_WIDGET_KEYS["learning_rate"], help="Optimizer step size. Smaller values update weights more cautiously; larger values can train faster but may become unstable.")
        c5.number_input("Maximum epochs", min_value=1, max_value=500, step=1, key=NEW_RUN_WIDGET_KEYS["max_epochs"], help="Maximum number of full passes over the training data. Training may stop earlier when the Patience threshold is reached.")

        max_epochs_value = max(1, int(st.session_state.get(NEW_RUN_WIDGET_KEYS["max_epochs"], 20)))
        if int(st.session_state.get(NEW_RUN_WIDGET_KEYS["patience"], 5)) > max_epochs_value:
            st.session_state[NEW_RUN_WIDGET_KEYS["patience"]] = max_epochs_value
        c6.number_input("Patience", min_value=1, max_value=max_epochs_value, step=1, key=NEW_RUN_WIDGET_KEYS["patience"], help="Stop after this many consecutive epochs without validation improvement.")

        live_config = _read_new_run_widget_values()
        st.session_state.training_config = dict(live_config)
        duplicate_live = find_visible_training_match(checkpoints, live_config) if checkpoints else None

        duplicate_name = str(duplicate_live.get("run_name", "")) if duplicate_live else ""
        force_duplicate = False
        if duplicate_live:
            st.warning(
                f"A trained checkpoint with the same visible new-run settings already exists: "
                f"{duplicate_name}. Change any value to create a different experiment, "
                f"or explicitly allow a duplicate training run."
            )
            duplicate_widget_key = _duplicate_checkbox_widget_key()
            force_duplicate = st.checkbox(
                "Train duplicate as a new run",
                key=duplicate_widget_key,
                help="Creates a new checkpoint with the same settings. Existing checkpoints are not overwritten.",
            )
            st.session_state.force_train_new_run = bool(force_duplicate)
            if force_duplicate:
                st.info("Duplicate training is enabled. A separate new checkpoint will be created only after Start training.")
            else:
                st.caption("Prepare is locked because these settings already match an existing trained checkpoint.")
        else:
            _clear_duplicate_checkbox_states()
            st.success("New training settings detected. Preparing this run will create a new checkpoint.")

        prepare_disabled = bool(duplicate_live and not bool(st.session_state.get("force_train_new_run", False)))
        st.caption("A new run name is generated automatically only when training starts. Existing checkpoints are never overwritten.")

        if st.button("Prepare separate new run", key="prepare_separate_new_run", type="primary", use_container_width=True, disabled=prepare_disabled):
            live_config = _read_new_run_widget_values()
            duplicate_after = find_visible_training_match(checkpoints, live_config) if checkpoints else None
            force_now = bool(st.session_state.get("force_train_new_run", False))
            if duplicate_after and not force_now:
                st.session_state.force_train_new_run = False
                st.error(
                    f"A trained checkpoint with the same visible new-run settings already exists: "
                    f"{duplicate_after['run_name']}. Change a value or tick Train duplicate as a new run."
                )
                save_flow()
            else:
                st.session_state.training_config = dict(live_config)
                st.session_state.force_train_new_run = bool(duplicate_after and force_now)
                st.session_state.new_run_config_source_checkpoint = selected if selected_info else st.session_state.get("new_run_config_source_checkpoint")
                st.session_state.new_run_block_reason = ""
                clear_training_view()
                st.session_state.training_mode = "new"
                st.session_state.resume_training_config = {}
                st.session_state.training_config["run_name"] = None
                set_stage_state("Model", "completed")
                st.session_state.stage = "Training"
                save_flow()
                st.rerun()


def _patience_from_history(history: list[dict]) -> int:
    count = 0
    for item in reversed(history):
        if item.get("status") == "new_best":
            break
        count += 1
    return count


def training_page() -> None:
    state = stage_state("Training")
    stage_header("Run Mamba policy", "Train the behavioral-cloning model using the selected dataset, cached embeddings, and hyperparameters.", state.upper(), tone_for(state))
    csv_path = st.session_state.filtered_csv or find_filtered_csv(st.session_state.dataset_root)
    if not csv_path or stage_state("Model") != "completed":
        st.warning("Prepare a model configuration first.")
        return

    run_mode = st.session_state.get("training_mode", "new")
    if run_mode == "resume":
        config = dict(st.session_state.get("resume_training_config") or {})
    else:
        config = dict(st.session_state.training_config)
    config["filtered_csv"] = csv_path
    execution_source = config.get("run_name") or st.session_state.current_training_run or "Generated when training starts"
    source_rows = [
        ("Run mode", "Resume interrupted training" if run_mode == "resume" else "New training run"),
        ("Checkpoint / run", execution_source),
        ("Configuration source", "Selected checkpoint config.json (locked for Resume)" if run_mode == "resume" else "Editable new-run settings prepared in Model"),
    ]
    if run_mode == "new" and st.session_state.get("force_train_new_run"):
        source_rows.append(("Retraining status", "Deliberate separate retraining with settings that match an existing checkpoint"))
    source_config_block(
        "Training execution source",
        "The active run source and the exact read-only configuration used by this training process.",
        source_rows,
        _saved_config_items(config),
    )
    lines = latest_lines("training") if state in {"running", "stopping", "stopped", "completed", "failed"} else []
    parsed = parse_training_progress(lines)
    patience_limit = int(parsed.get("patience") or config.get("patience", 5))
    patience_counter = int(parsed.get("patience_counter") or _patience_from_history(parsed.get("history", [])))
    best_epoch = parsed.get("best_epoch") or "--"
    best_val_loss = parsed.get("best_val_loss") if parsed.get("best_val_loss") is not None else "--"

    if state == "completed":
        progress_ratio = 1.0
        if parsed.get("early_stopping") or (parsed["max_epochs"] and parsed["epoch"] < parsed["max_epochs"]):
            progress_detail = f"Completed early at epoch {parsed['epoch']} / {parsed['max_epochs']} - early stopping"
        elif parsed["max_epochs"]:
            progress_detail = f"Completed at epoch {parsed['epoch']} / {parsed['max_epochs']}"
        else:
            progress_detail = "Training completed"
    else:
        progress_ratio = parsed["ratio"]
        progress_detail = f"Epoch {parsed['epoch']} / {parsed['max_epochs']}" if parsed["max_epochs"] else "Waiting to start"
    progress_card(progress_ratio, "Training progress", progress_detail)
    metrics([
        ("Current", state, "Mamba policy"),
        ("Train loss", parsed["train_loss"] if parsed["train_loss"] is not None else "--", "latest"),
        ("Val loss", parsed["val_loss"] if parsed["val_loss"] is not None else "--", "latest"),
        ("Best epoch", best_epoch, "validation"),
        ("Patience", f"{patience_counter} / {patience_limit}", "epochs without improvement"),
    ])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        training_loss_chart(parsed.get("history", []))
    with c2:
        training_status_chart(parsed.get("history", []))
    log_box(lines, "Waiting to start training...", title="Training log")

    checkpoints = discover_checkpoints()
    current_name = st.session_state.current_training_run or config.get("run_name") or ""
    current = checkpoints.get(current_name)
    if state == "running":
        st.button("Stop safely", on_click=request_stop, key="stop_training", use_container_width=True)
        st.caption("Resume starts from the last fully saved epoch checkpoint.")
    elif state == "stopping":
        st.button("Stopping safely...", disabled=True, use_container_width=True)
    elif state == "stopped":
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Back to model", key="back_training_stopped", use_container_width=True):
            remember_checkpoint(current_name)
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
        resume_path = current["resume_path"] if current and current.get("can_resume") else None
        if c2.button("Resume training", key="resume_training", type="primary", use_container_width=True, disabled=not bool(resume_path)):
            launch("training", build_train_command(config, resume_path=resume_path), "Training", mode="resume")
    elif state == "completed":
        st.success(f"Training completed. Best epoch: {best_epoch} - Best validation loss: {best_val_loss}")
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Train another configuration", key="another_training", use_container_width=True):
            st.session_state.training_mode = "new"
            clear_training_view()
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
        if c2.button("Evaluate best model", key="evaluate_trained", type="primary", use_container_width=True, disabled=not checkpoint_ready()):
            set_stage_state("Evaluation", "not_started")
            st.session_state.stage = "Evaluation"
            save_flow()
            st.rerun()
    elif state == "failed":
        duplicate_run = duplicate_checkpoint_from_lines(lines)
        if duplicate_run:
            st.warning(f"Training was not started because matching trained weights already exist: {duplicate_run}. Return to Model to evaluate them, change the new-run values, or explicitly force a retraining run.")
        if st.button("Back to model", key="failed_training_back", use_container_width=True):
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
    else:
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Back to model", key="back_training", use_container_width=True):
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
        if c2.button("Start training", key="start_training", type="primary", use_container_width=True):
            st.session_state.training_mode = "new"
            clear_stage_log("training")
            clear_evaluation_view()
            if not config.get("run_name"):
                config["run_name"] = suggest_run_name(config)
                st.session_state.training_config["run_name"] = config["run_name"]
            st.session_state.current_training_run = config["run_name"]
            st.session_state.last_used_checkpoint = config["run_name"]
            save_flow()
            launch("training", build_train_command(config, force_train=bool(st.session_state.get("force_train_new_run", False))), "Training", mode="new")

def _latest_prediction_example(parsed_metrics: dict[str, Any]) -> tuple[list[float], list[float]]:
    examples = parsed_metrics.get("prediction_examples") or []
    if not examples:
        return [], []
    current = examples[-1]
    return safe_float_values(current.get("ground_truth")), safe_float_values(current.get("prediction"))


def _evaluation_log_lines(lines: list[str]) -> list[str]:
    """Return the full Evaluation log for display.

    The UI keeps this log visible and complete, while parsing metrics from it separately.
    Empty or internal None values are removed so they are not displayed to the user.
    """
    cleaned: list[str] = []
    for line in lines or []:
        text = str(line).rstrip("\n")
        if text.strip() in {"", "None", "null"}:
            continue
        cleaned.append(text)
    return cleaned


def evaluation_page() -> None:
    state = stage_state("Evaluation")
    stage_header("Evaluate existing weights", "Run a selected checkpoint without changing model weights.", state.upper(), tone_for(state))
    csv_path = st.session_state.filtered_csv or find_filtered_csv(st.session_state.dataset_root)
    checkpoints = discover_checkpoints()
    checkpoint = checkpoints.get(st.session_state.selected_checkpoint or "")
    if not csv_path:
        st.warning("A filtered CSV is required.")
        return
    if not checkpoint or not checkpoint["has_model"]:
        st.warning("Select a trained checkpoint with model.pth in the Model page first.")
        return

    lines = latest_lines("evaluation", max_lines=None) if state in {"running", "stopping", "stopped", "completed", "failed"} else []
    parsed_metrics = parse_evaluation_metrics(lines)
    checkpoint_config = load_json(checkpoint.get("config_path", ""))
    ratio = 1.0 if state == "completed" else float(parsed_metrics.get("ratio") or 0.0)
    done_batches = int(parsed_metrics.get("done_batches") or 0)
    total_batches = int(parsed_metrics.get("total_batches") or 0)
    sample_count = int(parsed_metrics.get("samples") or 0)
    saved_examples = int(parsed_metrics.get("printed_examples") or 0)
    requested_examples = int(st.session_state.examples_to_display or 1)

    if state == "completed":
        progress_detail = f"Evaluation completed - {sample_count} samples processed - {saved_examples} prediction examples saved"
    elif state in {"running", "stopping"}:
        progress_detail = f"Evaluating validation data - {sample_count} samples processed"
    elif state == "stopped":
        progress_detail = f"Evaluation paused - {sample_count} samples processed"
    else:
        progress_detail = "Waiting to start evaluation"
    progress_card(ratio, "Evaluation progress", progress_detail)

    metrics([
        ("Current", state, "evaluation"),
        ("Samples", parsed_metrics.get("samples") or "--", "processed"),
        ("MAE", parsed_metrics.get("mae") if parsed_metrics.get("mae") is not None else "--", "running / final"),
        ("MSE", parsed_metrics.get("mse") if parsed_metrics.get("mse") is not None else "--", "running / final"),
        ("Saved examples", saved_examples if saved_examples else requested_examples, "for Results review"),
    ])

    source_config_block(
        "Evaluation source",
        "Evaluation always loads the selected trained weights and their saved configuration.",
        [
            ("Selected checkpoint", checkpoint["run_name"]),
            ("Configuration source", f"checkpoints/{checkpoint['run_name']}/config.json"),
            ("Prediction examples", f"{requested_examples} requested; {saved_examples or 0} saved so far"),
        ],
        _saved_config_items(checkpoint_config),
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        per_dimension_mae_chart(safe_float_values(parsed_metrics.get("per_dimension_mae")))
    with c2:
        gt, pred = _latest_prediction_example(parsed_metrics)
        prediction_comparison_chart(gt, pred)

    # Keep the evaluation log visible and complete, but do not cut it to two examples.
    # The log is scrollable inside the fixed log card, like Training.
    full_log_lines = _evaluation_log_lines(lines)
    log_box(full_log_lines, "Waiting to start evaluation...", title="Evaluation log", max_lines=None)

    if state == "running":
        st.button("Cancel evaluation", on_click=request_stop, key="stop_evaluation", use_container_width=True)
    elif state == "stopping":
        st.button("Cancelling evaluation...", disabled=True, use_container_width=True)
    elif state == "completed":
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Back to evaluation settings", key="back_to_evaluation_settings", use_container_width=True):
            set_stage_state("Evaluation", "not_started")
            save_flow()
            st.rerun()
        if c2.button("View results", key="view_results", type="primary", use_container_width=True):
            st.session_state.latest_eval_metrics = parsed_metrics
            # Always open the Results player from the first saved example after a new Evaluation run.
            st.session_state["results_example_index"] = 0
            st.session_state["results_example_signature"] = ""
            set_stage_state("Results", "completed")
            st.session_state.stage = "Results"
            save_flow()
            st.rerun()
    elif state == "stopped":
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Back to model", key="back_evaluation_stopped", use_container_width=True):
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
        if c2.button("Back to evaluation settings", key="rerun_evaluation_stopped", type="primary", use_container_width=True):
            set_stage_state("Evaluation", "not_started")
            save_flow()
            st.rerun()
    elif state == "failed":
        if st.button("Back to model", key="failed_evaluation_back", use_container_width=True):
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
    else:
        max_examples = checkpoint_max_preview_examples(checkpoint)
        current_examples = min(max_examples, max(1, int(st.session_state.examples_to_display)))
        examples = st.number_input(
            "Prediction examples to save for review",
            min_value=1,
            max_value=max_examples,
            value=current_examples,
            step=1,
            key="evaluation_examples_number",
            help="Evaluation always runs on the complete validation split. This value controls only how many prediction examples are stored for review in Results.",
        )
        st.caption(f"Evaluation always runs on the full validation set. Choose any number of prediction examples from 1 to {max_examples:,}; this controls only how many predictions are saved for review in Results.")
        st.session_state.evaluation_mode = "Full"
        st.session_state.examples_to_display = int(examples)
        save_flow()
        action_spacer()
        c3, c4 = st.columns(2, gap="large")
        if c3.button("Back to model", key="back_evaluation_before_start", use_container_width=True):
            st.session_state.stage = "Model"
            save_flow()
            st.rerun()
        if c4.button("Start full evaluation", key="start_evaluation", type="primary", use_container_width=True):
            launch("evaluation", build_evaluate_command(checkpoint, csv_path, "Full", int(examples)), "Evaluation", mode="new")


def _find_visual_frames(dataset_root: str | None, limit: int = 400) -> list[str]:
    """Return a small ordered set of image frames for the Results playback panel.

    Evaluation logs currently contain action vectors but not exact image paths.
    Until evaluate.py is extended to save the frame path per example, the UI
    shows a visual context sequence from the selected raw dataset folder.
    """
    if not dataset_root:
        return []
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    frames: list[str] = []
    preferred_tokens = ("image", "images", "rgb", "camera", "frame")
    try:
        for path in root.rglob("*"):
            if len(frames) >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            parts = "/".join(part.lower() for part in path.parts)
            if any(token in parts for token in preferred_tokens):
                frames.append(str(path))
        if not frames:
            for path in root.rglob("*"):
                if len(frames) >= limit:
                    break
                if path.is_file() and path.suffix.lower() in extensions:
                    frames.append(str(path))
    except Exception:
        return []
    return sorted(frames)


def _image_data_uri(frame_path: str) -> str | None:
    """Convert a local image into a browser-safe data URI for the styled Results player."""
    try:
        path = Path(frame_path)
        if not path.exists() or not path.is_file():
            return None
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


def _short_frame_path(frame_path: str | None, keep_parts: int = 5) -> tuple[str, str]:
    """Return the full frame path for Results display."""
    if not frame_path:
        return "No frame path available", ""
    full = str(frame_path)
    return full, full


def _display_camera_frame(frame_path: str | None, title: str, subtitle: str, badge: str | None = None) -> None:
    """Compact Figma-style camera player card for Results."""
    safe_title = html.escape(str(title))
    safe_subtitle = html.escape(str(subtitle))
    safe_badge = html.escape(str(badge)) if badge else ""
    short_path, full_path = _short_frame_path(frame_path)
    safe_short_path = html.escape(short_path)
    safe_full_path = html.escape(full_path)

    header_badge = f'<div class="example-pill player-badge">{safe_badge}</div>' if safe_badge else ""

    if frame_path and Path(frame_path).exists():
        data_uri = _image_data_uri(frame_path)
        if data_uri:
            st.markdown(
                f"""
                <div class="sequence-player-shell compact-player result-camera-card">
                  <div class="player-card-head">
                    <div class="player-title-wrap">
                      <div class="section-title">{safe_title}</div>
                      <div class="section-sub">{safe_subtitle}</div>
                    </div>
                    {header_badge}
                  </div>
                  <div class="camera-frame-card compact-frame">
                    <img src="{data_uri}" alt="Robot camera frame" />
                  </div>
                  <div class="frame-path-caption full-path" title="{safe_full_path}">{safe_short_path}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    st.markdown(
        f"""
        <div class="sequence-player-shell compact-player result-camera-card">
          <div class="player-card-head">
            <div class="player-title-wrap">
              <div class="section-title">{safe_title}</div>
              <div class="section-sub">{safe_subtitle}</div>
            </div>
            {header_badge}
          </div>
          <div class="camera-placeholder compact-camera">
            <div class="camera-title">Robot camera frame</div>
            <div class="camera-subtitle">No frame path was saved for this prediction example yet</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _action_table_card(rows: list[dict[str, Any]]) -> None:
    """Render the pose/gripper table inside a styled results card."""
    header_cells = ["Value", "Model", "Ground truth", "Abs. error"]
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(col, '--')))}</td>"
            for col in header_cells
        )
        body_rows.append(f"<tr>{cells}</tr>")
    table_html = "".join(body_rows)
    st.markdown(
        f"""
        <div class="action-table-card">
          <div class="action-table-head">
            <div>
              <div class="section-title">Pose + gripper action table</div>
              <div class="section-sub">Current prediction vector compared with ground truth and absolute error.</div>
            </div>
          </div>
          <div class="action-table-wrap">
            <table class="action-results-table">
              <thead>
                <tr><th>Value</th><th>Model</th><th>Ground truth</th><th>Abs. error</th></tr>
              </thead>
              <tbody>{table_html}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _prediction_sequence_viewer(result: dict[str, Any], key_prefix: str = "results") -> None:
    """Display saved prediction examples one at a time with frame context and action table."""
    examples = result.get("prediction_examples") or []
    if not examples:
        st.info("No saved prediction examples are available yet. Run Evaluation and choose how many examples to save for review.")
        return

    total = len(examples)
    index_key = f"{key_prefix}_example_index"
    signature_key = f"{key_prefix}_example_signature"

    signature = f"{st.session_state.get('selected_checkpoint', '')}|{result.get('samples')}|{result.get('mae')}|{total}"
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[index_key] = 0

    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    st.session_state[index_key] = max(0, min(int(st.session_state[index_key]), total - 1))

    current_idx = int(st.session_state[index_key])
    current = examples[current_idx]

    gt = safe_float_values(current.get("ground_truth"))
    pred = safe_float_values(current.get("prediction"))

    max_len = max(len(gt), len(pred), 7)
    labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    while len(labels) < max_len:
        labels.append(f"dim {len(labels) + 1}")

    exact_frame_path = current.get("frame_path")
    frames = _find_visual_frames(st.session_state.get("dataset_root"), limit=500)

    frame_path = exact_frame_path
    frame_number = None

    if not frame_path and frames:
        frame_idx = int(round(current_idx * (len(frames) - 1) / max(total - 1, 1)))
        frame_idx = max(0, min(frame_idx, len(frames) - 1))
        frame_path = frames[frame_idx]
        frame_number = frame_idx + 1

    metadata_bits = [
        f"Source log example: {current.get('index', current_idx + 1)}",
        f"Example MAE: {current.get('mae', '--')}",
    ]

    if current.get("trajectory_id") not in {None, "", "None"}:
        metadata_bits.append(f"Trajectory: {current.get('trajectory_id')}")

    if current.get("frame_index") not in {None, "", "None"}:
        metadata_bits.append(f"Frame index: {current.get('frame_index')}")

    if current.get("dataset_index") not in {None, "", "None"}:
        metadata_bits.append(f"Dataset window: {current.get('dataset_index')}")

    if exact_frame_path:
        frame_subtitle = "Exact frame saved by Evaluation - " + " - ".join(metadata_bits)
    else:
        frame_subtitle = (
            f"Context frame {frame_number} of {len(frames)} - "
            if frames
            else "Visual frame placeholder - "
        )
        frame_subtitle += "exact frame path was not available in this Evaluation log."

    rows = []
    for i in range(max_len):
        model_value = pred[i] if i < len(pred) else None
        gt_value = gt[i] if i < len(gt) else None

        abs_error = (
            abs(model_value - gt_value)
            if model_value is not None and gt_value is not None
            else None
        )

        rows.append({
            "Value": labels[i],
            "Model": "--" if model_value is None else f"{model_value:.6f}",
            "Ground truth": "--" if gt_value is None else f"{gt_value:.6f}",
            "Abs. error": "--" if abs_error is None else f"{abs_error:.6f}",
        })

    with st.container(key="results_unified_frame"):
        _display_camera_frame(
            frame_path,
            "Prediction example",
            frame_subtitle,
            badge=None,
        )

        first_col, prev_col, progress_col, next_col, last_col = st.columns(
            [1, 1, 1, 1, 1],
            gap="medium"
        )

        if first_col.button(
            "⏮ First",
            key=f"{key_prefix}_first_example",
            use_container_width=True,
            disabled=current_idx <= 0,
        ):
            st.session_state[index_key] = 0
            st.rerun()

        if prev_col.button(
            "‹ Previous",
            key=f"{key_prefix}_prev_example",
            use_container_width=True,
            disabled=current_idx <= 0,
        ):
            st.session_state[index_key] = max(0, current_idx - 1)
            st.rerun()

        progress_pct = ((current_idx + 1) / total) * 100 if total else 0

        progress_col.markdown(
            f"""
            <div class="player-progress-slot">
              <div class="results-progress-center">
                <div class="example-counter">Example {current_idx + 1} / {total}</div>
                <div class="results-progress-track">
                  <div class="results-progress-fill" style="width:{progress_pct:.2f}%"></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if next_col.button(
            "Next ›",
            key=f"{key_prefix}_next_example",
            use_container_width=True,
            disabled=current_idx >= total - 1,
        ):
            st.session_state[index_key] = min(total - 1, current_idx + 1)
            st.rerun()

        if last_col.button(
            "Last ⏭",
            key=f"{key_prefix}_last_example",
            use_container_width=True,
            disabled=current_idx >= total - 1,
        ):
            st.session_state[index_key] = total - 1
            st.rerun()

        st.markdown('<div class="results-controls-separator"></div>', unsafe_allow_html=True)

        _action_table_card(rows)

def results_page() -> None:
    stage_header("Results and playback review", "Review metrics, browse saved prediction examples, and inspect action-level errors.", "RESULTS", "green")
    result = st.session_state.latest_eval_metrics or {}
    if not result:
        eval_lines = latest_lines("evaluation", max_lines=None)
        result = parse_evaluation_metrics(eval_lines) if eval_lines else {}
    checkpoints = discover_checkpoints()
    checkpoint = checkpoints.get(st.session_state.selected_checkpoint or "")
    checkpoint_config = load_json(checkpoint.get("config_path", "")) if checkpoint else {}
    examples = result.get("prediction_examples") or []

    metrics([
        ("Evaluated samples", result.get("samples") or "--", "validation windows"),
        ("MSE", result.get("mse") if result.get("mse") is not None else "--", "overall"),
        ("MAE", result.get("mae") if result.get("mae") is not None else "--", "overall"),
        ("Saved examples", len(examples) or result.get("printed_examples") or "--", "prediction previews"),
    ])

    source_config_block(
        "Results source",
        "Metrics and saved prediction examples produced by the latest Evaluation run.",
        [
            ("Checkpoint", (checkpoint or {}).get("run_name", "--")),
            ("Saved prediction examples", len(examples) or result.get("printed_examples") or "--"),
        ],
        _saved_config_items(checkpoint_config),
    )

    left, right = st.columns([1, 1], gap="small")
    with left:
        per_dimension_mae_chart(safe_float_values(result.get("per_dimension_mae")))
    with right:
        gt, pred = _latest_prediction_example(result)
        prediction_comparison_chart(gt, pred)

    _prediction_sequence_viewer(result, key_prefix="results")

    eval_lines = latest_lines("evaluation", max_lines=None)
    if eval_lines:
        log_box(_evaluation_log_lines(eval_lines), "No evaluation log available.", title="Evaluation log", max_lines=None)

    c1, c2 = st.columns(2, gap="large")
    if c1.button("Back to evaluation settings", key="back_results", use_container_width=True):
        st.session_state.stage = "Evaluation"
        set_stage_state("Evaluation", "not_started")
        save_flow()
        st.rerun()
    if c2.button("Choose new dataset", key="results_back_to_dataset", type="primary", use_container_width=True):
        reset_pipeline_view()
        st.rerun()


PAGE_RENDERERS = {
    "Dataset": dataset_page,
    "YOLO": yolo_page,
    "OpenCLIP": openclip_page,
    "Model": model_page,
    "Training": training_page,
    "Evaluation": evaluation_page,
    "Results": results_page,
}


def render_stage_body() -> None:
    PAGE_RENDERERS[st.session_state.stage]()


init_state()
if refresh_process():
    st.rerun()

# Keep the compact header outside the live fragment.  Re-rendering the entire
# page every second caused a visible vertical jump during YOLO stop/resume.
# The body remains live while a process runs; a full rerun occurs only when the
# process actually changes state, which updates the stepper as well.
page_shell()
if active_process() and hasattr(st, "fragment"):
    @st.fragment(run_every=1.0)
    def live_stage_fragment() -> None:
        # Render the final STOPPED / COMPLETED state inside the same fragment.
        # A full-page rerun during process exit caused the YOLO screen to jump.
        refresh_process()
        render_stage_body()

    live_stage_fragment()
else:
    render_stage_body()
