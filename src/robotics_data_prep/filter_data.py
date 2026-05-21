import os
import csv
import glob
import pickle
import argparse
import numpy as np
from ultralytics import YOLO

def check_overlap(box1_xyxy, box2_xyxy):
    """
    Checks if there is a geometric overlap between two bounding boxes.
    
    Args:
        box1_xyxy (list): [x1, y1, x2, y2] of the first box.
        box2_xyxy (list): [x1, y1, x2, y2] of the second box.
        
    Returns:
        bool: True if boxes overlap, False otherwise.
    """
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    return (x2 > x1 and y2 > y1)

def main():
    # --- 1. Command Line Arguments Setup ---
    parser = argparse.ArgumentParser(description="Filter robotic pick trajectories using Sensor Fusion (Kinematics + YOLO Vision).")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the raw trajectories dataset directory.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the YOLO model weights (.pt file).")
    parser.add_argument('--threshold', type=float, default=0.0755, help="Motor value threshold for determining a stable lift.")
    parser.add_argument('--output', type=str, default='sensor_fusion_report.csv', help="Output CSV filename.")
    args = parser.parse_args()

    print("[INFO] Loading Computer Vision model (YOLO)...")
    model = YOLO(args.weights)

    print(f"[INFO] Scanning directory: {args.data_dir}")

    # --- 2. Collect Trajectories ---
    all_trajs = []
    for root, dirs, files in os.walk(args.data_dir):
        for d in dirs:
            if d.startswith('traj'):
                all_trajs.append(os.path.join(root, d))

    all_trajs = sorted(all_trajs)
    total_folders = len(all_trajs)

    if total_folders == 0:
        print("[ERROR] No trajectory folders found. Exiting.")
        exit(1)

    print(f"[INFO] Found {total_folders} trajectories. Starting analysis (Threshold = {args.threshold})...")
    results_summary = []

    # --- 3. Main Processing Loop ---
    for idx, traj_folder in enumerate(all_trajs):
        parent_folder = os.path.basename(os.path.dirname(traj_folder))
        base_traj_name = os.path.basename(traj_folder)
        full_traj_name = f"{parent_folder}/{base_traj_name}"

        obs_path = os.path.join(traj_folder, 'obs_dict.pkl')
        if not os.path.exists(obs_path):
            continue

        try:
            # Filter incomplete trajectories
            image_files = sorted(glob.glob(os.path.join(traj_folder, "**", "*.jpg"), recursive=True))
            if len(image_files) < 35:
                continue

            with open(obs_path, 'rb') as f:
                obs_data = pickle.load(f)

            gripper_states = obs_data['state'][:, -1]
            z_values = obs_data['eef_transform'][:, 2, 3]

            # Identify key frames: grasp moment (lowest Z) and peak lift (highest Z)
            half_way = len(z_values) // 2
            grasp_idx = int(np.argmin(z_values[:half_way]))
            peak_idx = grasp_idx + int(np.argmax(z_values[grasp_idx:]))
            
            # --- Phase A: Kinematic Stability Check ---
            lift_segment = gripper_states[grasp_idx + 5 : peak_idx]
            
            if len(lift_segment) > 0:
                stable_val = np.min(lift_segment)
            else:
                stable_val = gripper_states[peak_idx]

            # Apply strict calibrated threshold to prevent mid-air drops classification as success
            if stable_val < args.threshold:
                status = "FAILURE"
                reason = "Dropped mid-air or Grabbed empty air"
            elif stable_val > 0.85:
                status = "FAILURE"
                reason = "Missed! Gripper stayed wide open"
            else:
                # --- Phase B: Vision Verification (YOLO) ---
                results = model(image_files[grasp_idx], verbose=False)[0].boxes
                current_gripper = None
                saw_target = False

                for box in results:
                    if model.names[int(box.cls[0])] == 'gripper':
                        current_gripper = box.xyxy[0].tolist()
                        break

                if current_gripper is not None:
                    for box in results:
                        if model.names[int(box.cls[0])] == 'object':
                            if check_overlap(current_gripper, box.xyxy[0].tolist()):
                                saw_target = True
                                break

                status = "SUCCESS"
                reason = "Stable Lift (Motor + Vision)" if saw_target else "Blind Lift (Motor stable, Vision obscured)"

            results_summary.append({
                'Trajectory': full_traj_name,
                'Full_Path': os.path.abspath(traj_folder),
                'Status': status,
                'Stable_Motor_Value': round(float(stable_val), 4),
                'Reason': reason
            })

            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"[PROGRESS] Processed {idx + 1}/{total_folders} trajectories.")

        except Exception as e:
            print(f"[WARNING] Skipping {full_traj_name} due to processing error: {e}")
            continue

    # --- 4. Export Results ---
    csv_filename = os.path.join(args.data_dir, args.output)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Trajectory', 'Full_Path', 'Status', 'Stable_Motor_Value', 'Reason'])
        writer.writeheader()
        writer.writerows(results_summary)

    # Print Summary
    if results_summary:
        success_count = sum(1 for r in results_summary if r['Status'] == 'SUCCESS')
        acc = (success_count / len(results_summary)) * 100
        print("\n=== RUN SUMMARY ===")
        print(f"Total Processed Trajectories: {len(results_summary)}")
        print(f"Total Pick Successes: {success_count}")
        print(f"Overall Success Rate: {acc:.1f}%")
        print(f"[INFO] Report saved successfully to: {csv_filename}")

if __name__ == '__main__':
    main()