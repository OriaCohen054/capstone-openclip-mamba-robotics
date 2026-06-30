import argparse
import csv
import glob
import os
import pickle
import signal
import sys
from typing import Dict, List, Set

import numpy as np
from ultralytics import YOLO


CSV_FIELDNAMES = [
    "Trajectory",
    "Full_Path",
    "Status",
    "Stable_Motor_Value",
    "Reason",
]


def check_overlap(box1_xyxy, box2_xyxy):
    """
    Check whether two bounding boxes overlap.
    """
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    return x2 > x1 and y2 > y1


def normalize_path(path_value: str) -> str:
    """
    Normalize a path for stable comparisons during resume.
    """
    return os.path.abspath(os.path.normpath(path_value))


def collect_trajectory_folders(data_dir: str) -> List[str]:
    """
    Recursively collect trajectory folders whose names start with 'traj'.
    """
    all_trajs = []

    for root, dirs, _ in os.walk(data_dir):
        for directory_name in dirs:
            if directory_name.startswith("traj"):
                all_trajs.append(
                    normalize_path(os.path.join(root, directory_name))
                )

    return sorted(all_trajs)


def load_existing_rows(csv_filename: str) -> List[Dict[str, str]]:
    """
    Load previously saved CSV rows.

    This allows a stopped filtering run to continue without recomputing
    trajectories that were already processed.
    """
    if not os.path.exists(csv_filename):
        return []

    with open(csv_filename, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            return []

        missing_columns = [
            column_name
            for column_name in CSV_FIELDNAMES
            if column_name not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "Existing CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        return list(reader)


def get_processed_paths(existing_rows: List[Dict[str, str]]) -> Set[str]:
    """
    Build a set of trajectory paths that were already saved to the CSV.
    """
    processed_paths = set()

    for row in existing_rows:
        full_path = str(row.get("Full_Path", "")).strip()

        if full_path:
            processed_paths.add(normalize_path(full_path))

    return processed_paths


def append_result_row(
    writer: csv.DictWriter,
    csv_file,
    row: Dict[str, object],
) -> None:
    """
    Append one result row and force it to disk immediately.

    Saving after every trajectory makes interruption and resume safe.
    """
    writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())


def build_result_row(
    traj_folder: str,
    status: str,
    reason: str,
    stable_motor_value="",
) -> Dict[str, object]:
    """
    Build one CSV result row.
    """
    parent_folder = os.path.basename(os.path.dirname(traj_folder))
    base_traj_name = os.path.basename(traj_folder)
    full_traj_name = f"{parent_folder}/{base_traj_name}"

    return {
        "Trajectory": full_traj_name,
        "Full_Path": normalize_path(traj_folder),
        "Status": status,
        "Stable_Motor_Value": stable_motor_value,
        "Reason": reason,
    }


def process_trajectory(
    model,
    traj_folder: str,
    threshold: float,
) -> Dict[str, object]:
    """
    Analyze one trajectory using gripper-state logic and YOLO verification.

    Returns:
        A CSV row with SUCCESS, FAILURE, or SKIPPED status.
    """
    obs_path = os.path.join(traj_folder, "obs_dict.pkl")

    if not os.path.exists(obs_path):
        return build_result_row(
            traj_folder=traj_folder,
            status="SKIPPED",
            reason="Missing obs_dict.pkl",
        )

    image_files = sorted(
        glob.glob(
            os.path.join(traj_folder, "**", "*.jpg"),
            recursive=True,
        )
    )

    if len(image_files) < 35:
        return build_result_row(
            traj_folder=traj_folder,
            status="SKIPPED",
            reason="Incomplete trajectory: fewer than 35 JPG images",
        )

    with open(obs_path, "rb") as pickle_file:
        obs_data = pickle.load(pickle_file)

    gripper_states = obs_data["state"][:, -1]
    z_values = obs_data["eef_transform"][:, 2, 3]

    # Identify key frames: grasp moment (lowest Z) and peak lift (highest Z).
    half_way = len(z_values) // 2
    grasp_idx = int(np.argmin(z_values[:half_way]))
    peak_idx = grasp_idx + int(np.argmax(z_values[grasp_idx:]))

    # Phase A: Kinematic stability check.
    lift_segment = gripper_states[grasp_idx + 5:peak_idx]

    if len(lift_segment) > 0:
        stable_val = np.min(lift_segment)
    else:
        stable_val = gripper_states[peak_idx]

    stable_motor_value = round(float(stable_val), 4)

    if stable_val < threshold:
        return build_result_row(
            traj_folder=traj_folder,
            status="FAILURE",
            stable_motor_value=stable_motor_value,
            reason="Dropped mid-air or Grabbed empty air",
        )

    if stable_val > 0.85:
        return build_result_row(
            traj_folder=traj_folder,
            status="FAILURE",
            stable_motor_value=stable_motor_value,
            reason="Missed! Gripper stayed wide open",
        )

    # Phase B: Vision verification using YOLO.
    results = model(image_files[grasp_idx], verbose=False)[0].boxes

    current_gripper = None
    saw_target = False

    for box in results:
        if model.names[int(box.cls[0])] == "gripper":
            current_gripper = box.xyxy[0].tolist()
            break

    if current_gripper is not None:
        for box in results:
            if model.names[int(box.cls[0])] == "object":
                if check_overlap(
                    current_gripper,
                    box.xyxy[0].tolist(),
                ):
                    saw_target = True
                    break

    reason = (
        "Stable Lift (Motor + Vision)"
        if saw_target
        else "Blind Lift (Motor stable, Vision obscured)"
    )

    return build_result_row(
        traj_folder=traj_folder,
        status="SUCCESS",
        stable_motor_value=stable_motor_value,
        reason=reason,
    )


def print_summary(csv_filename: str) -> None:
    """
    Print a summary based on all rows currently saved in the CSV.
    """
    rows = load_existing_rows(csv_filename)

    success_count = sum(
        1 for row in rows
        if row.get("Status") == "SUCCESS"
    )

    failure_count = sum(
        1 for row in rows
        if row.get("Status") == "FAILURE"
    )

    skipped_count = sum(
        1 for row in rows
        if row.get("Status") == "SKIPPED"
    )

    classified_count = success_count + failure_count

    if classified_count > 0:
        success_rate = success_count / classified_count * 100
    else:
        success_rate = 0.0

    print("\n=== RUN SUMMARY ===")
    print(f"Rows saved to CSV: {len(rows)}")
    print(f"Total Pick Successes: {success_count}")
    print(f"Total Failures: {failure_count}")
    print(f"Skipped Trajectories: {skipped_count}")
    print(f"Success Rate (excluding skipped): {success_rate:.1f}%")
    print(f"[INFO] Report saved to: {csv_filename}")


def handle_stop_signal(_signal_number, _frame) -> None:
    """
    Convert a termination signal into a safe interruption.

    This allows a UI process manager to stop the script gracefully.
    """
    raise KeyboardInterrupt


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Filter robotic pick trajectories using sensor fusion "
            "(kinematics + YOLO vision)."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the raw trajectories dataset directory.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the YOLO model weights (.pt file).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0755,
        help="Motor value threshold for determining a stable lift.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sensor_fusion_report.csv",
        help="Output CSV filename.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Start filtering from the beginning and overwrite an existing CSV. "
            "Without this flag, the script resumes automatically."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    """
    args = parse_args()

    # SIGTERM is commonly sent by UI process managers when the user clicks Stop.
    signal.signal(signal.SIGTERM, handle_stop_signal)

    csv_filename = os.path.join(args.data_dir, args.output)
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    if args.restart and os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"[INFO] Restart requested. Removed old CSV: {csv_filename}")

    existing_rows = load_existing_rows(csv_filename)
    processed_paths = get_processed_paths(existing_rows)

    if existing_rows:
        print(
            f"[INFO] Resume mode: found {len(existing_rows)} saved rows. "
            "Previously processed trajectories will be skipped."
        )
    else:
        print("[INFO] Starting a new filtering run.")

    print("[INFO] Loading Computer Vision model (YOLO)...")
    model = YOLO(args.weights)

    print(f"[INFO] Scanning directory: {args.data_dir}")
    all_trajs = collect_trajectory_folders(args.data_dir)
    total_folders = len(all_trajs)

    if total_folders == 0:
        print("[ERROR] No trajectory folders found. Exiting.")
        return 1

    print(
        f"[INFO] Found {total_folders} trajectories. "
        f"Starting analysis (Threshold = {args.threshold})..."
    )

    csv_exists = os.path.exists(csv_filename)
    csv_is_empty = (
        not csv_exists
        or os.path.getsize(csv_filename) == 0
    )

    completed_count = len(processed_paths)

    try:
        with open(
            csv_filename,
            mode="a",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDNAMES,
            )

            if csv_is_empty:
                writer.writeheader()
                csv_file.flush()
                os.fsync(csv_file.fileno())

            for traj_folder in all_trajs:
                normalized_traj_path = normalize_path(traj_folder)

                if normalized_traj_path in processed_paths:
                    continue

                try:
                    row = process_trajectory(
                        model=model,
                        traj_folder=traj_folder,
                        threshold=args.threshold,
                    )

                except Exception as error:
                    row = build_result_row(
                        traj_folder=traj_folder,
                        status="SKIPPED",
                        reason=f"Processing error: {error}",
                    )

                append_result_row(
                    writer=writer,
                    csv_file=csv_file,
                    row=row,
                )

                processed_paths.add(normalized_traj_path)
                completed_count += 1

                if completed_count % 50 == 0 or completed_count == total_folders:
                    print(
                        f"[PROGRESS] Saved {completed_count}/"
                        f"{total_folders} trajectories."
                    )

    except KeyboardInterrupt:
        print("\n[INFO] Stop requested.")
        print("[INFO] Progress was saved safely.")
        print("[INFO] Run the same command again to resume automatically.")
        print_summary(csv_filename)
        return 130

    print_summary(csv_filename)
    print("[INFO] Filtering completed successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
