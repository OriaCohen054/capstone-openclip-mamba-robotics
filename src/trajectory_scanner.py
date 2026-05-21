import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TrajectoryScanResult:
    """
    Represents one detected trajectory folder.

    Important:
    This scanner only checks folder structure.
    It does NOT decide whether the robot succeeded in the PNP task.

    Success/failure filtering will be implemented later, possibly using YOLO.
    """
    name: str
    path: str
    relative_path: str
    path_parts: List[str]
    is_valid_structure: bool
    reason: str
    num_images: int = 0


def count_image_files(images_dir: str) -> int:
    """
    Counts image files inside the images0 folder.
    """
    if not os.path.isdir(images_dir):
        return 0

    valid_extensions = (".jpg", ".jpeg", ".png")

    return len([
        file_name for file_name in os.listdir(images_dir)
        if file_name.lower().endswith(valid_extensions)
    ])


def is_valid_trajectory_folder(traj_path: str, base_dir: str) -> TrajectoryScanResult:
    """
    Checks whether a folder has the expected trajectory structure.

    A structurally valid trajectory folder should contain:
    - obs_dict.pkl
    - policy_out.pkl
    - images0 folder

    This function does NOT check whether the PNP task succeeded.
    """
    traj_name = os.path.basename(traj_path)

    relative_path = os.path.relpath(traj_path, base_dir)
    path_parts = relative_path.split(os.sep)

    obs_path = os.path.join(traj_path, "obs_dict.pkl")
    policy_path = os.path.join(traj_path, "policy_out.pkl")
    images_dir = os.path.join(traj_path, "images0")

    if not os.path.exists(obs_path):
        return TrajectoryScanResult(
            name=traj_name,
            path=traj_path,
            relative_path=relative_path,
            path_parts=path_parts,
            is_valid_structure=False,
            reason="Missing obs_dict.pkl"
        )

    if not os.path.exists(policy_path):
        return TrajectoryScanResult(
            name=traj_name,
            path=traj_path,
            relative_path=relative_path,
            path_parts=path_parts,
            is_valid_structure=False,
            reason="Missing policy_out.pkl"
        )

    if not os.path.isdir(images_dir):
        return TrajectoryScanResult(
            name=traj_name,
            path=traj_path,
            relative_path=relative_path,
            path_parts=path_parts,
            is_valid_structure=False,
            reason="Missing images0 folder"
        )

    image_count = count_image_files(images_dir)

    return TrajectoryScanResult(
        name=traj_name,
        path=traj_path,
        relative_path=relative_path,
        path_parts=path_parts,
        is_valid_structure=True,
        reason="Valid trajectory structure",
        num_images=image_count
    )


def scan_trajectory_folders(base_dir: str) -> List[TrajectoryScanResult]:
    """
    Recursively scans the dataset root folder and finds valid trajectory folders.

    Current behavior:
    - Finds folders that contain obs_dict.pkl, policy_out.pkl and images0.
    - Keeps full paths to avoid duplicate trajectory names.
    - Does NOT determine whether the PNP task succeeded.

    Future behavior:
    - Add YOLO-based visual success validation.
    - Add OpenCLIP-based visual validation if needed.
    - Add reliable data-based success metrics if available.
    """
    results = []

    if not os.path.isdir(base_dir):
        return results

    for root, dirs, files in os.walk(base_dir):
        has_obs = "obs_dict.pkl" in files
        has_policy = "policy_out.pkl" in files
        has_images = "images0" in dirs

        # Only folders that contain all required trajectory files are considered trajectories.
        if has_obs and has_policy and has_images:
            result = is_valid_trajectory_folder(root, base_dir)
            results.append(result)

    results.sort(key=lambda item: item.relative_path)

    return results


def get_valid_trajectory_options(results: List[TrajectoryScanResult]) -> Dict[str, str]:
    """
    Converts valid trajectory scan results into options for a Streamlit selectbox.

    The key is the relative path, not only the folder name.
    This prevents collisions when multiple trajectories have the same name.

    Returns:
        {
            "relative/path/to/traj13": "/full/path/to/traj13"
        }
    """
    options = {}

    for result in results:
        if result.is_valid_structure:
            options[result.relative_path] = result.path

    return options