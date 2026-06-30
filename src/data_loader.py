import csv
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip


def natural_sort_key(text: str) -> List[Any]:
    """
    Create a natural sorting key for file names.

    This prevents ordering issues such as:
        im_10.jpg appearing before im_2.jpg
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def find_valid_trajectory_folders(data_root: str) -> List[str]:
    """
    Recursively find valid trajectory folders.

    A valid trajectory folder must contain:
        - obs_dict.pkl
        - policy_out.pkl
        - images0/

    Args:
        data_root: Root directory containing trajectory folders.

    Returns:
        A sorted list of absolute trajectory folder paths.
    """
    trajectory_folders = []

    if not os.path.isdir(data_root):
        return trajectory_folders

    for root, dirs, files in os.walk(data_root):
        has_obs = "obs_dict.pkl" in files
        has_policy = "policy_out.pkl" in files
        has_images = "images0" in dirs

        if has_obs and has_policy and has_images:
            trajectory_folders.append(os.path.abspath(root))

    trajectory_folders.sort()
    return trajectory_folders


def load_trajectory_paths_from_csv(
    csv_path: str,
    status_filter: Optional[str] = "SUCCESS",
) -> List[str]:
    """
    Load trajectory paths from a filtered CSV file.

    The expected CSV is the output of robotics_data_prep/filter_data.py.
    It should contain a Full_Path column and optionally a Status column.

    Args:
        csv_path: Path to the CSV file.
        status_filter: If provided, only rows with this Status value are used.
                       Use None to load all rows.

    Returns:
        A sorted list of absolute trajectory folder paths.
    """
    trajectory_paths = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        if "Full_Path" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'Full_Path' column.")

        for row in reader:
            status = str(row.get("Status", "")).strip()
            full_path = str(row.get("Full_Path", "")).strip()

            if not full_path:
                continue

            if status_filter is not None and status != status_filter:
                continue

            trajectory_paths.append(os.path.abspath(full_path))

    trajectory_paths = sorted(list(set(trajectory_paths)))
    return trajectory_paths


def list_image_files(images_dir: str) -> List[str]:
    """
    List image files from a trajectory images directory.

    Args:
        images_dir: Path to the images0 directory.

    Returns:
        A naturally sorted list of image file names.
    """
    if not os.path.isdir(images_dir):
        return []

    valid_extensions = (".jpg", ".jpeg", ".png")

    image_files = [
        file_name
        for file_name in os.listdir(images_dir)
        if file_name.lower().endswith(valid_extensions)
    ]

    image_files.sort(key=natural_sort_key)
    return image_files


def get_embedding_file_name(
    clip_model_name: str,
    clip_pretrained: str,
) -> str:
    """
    Build the expected embedding filename.

    This must match the filename created by openclip_embed.py.

    Args:
        clip_model_name: OpenCLIP model name.
        clip_pretrained: OpenCLIP pretrained weights name.

    Returns:
        Expected .npy embedding filename.
    """
    safe_model = clip_model_name.replace("/", "_").replace("-", "").lower()
    safe_pretrained = clip_pretrained.replace("/", "_").replace("-", "_").lower()

    return f"openclip_{safe_model}_{safe_pretrained}_embeddings.npy"


def load_policy_actions(policy_path: str) -> np.ndarray:
    """
    Load actions from policy_out.pkl.

    The file may contain either:
        - a list/array of actions
        - a list of dictionaries with keys such as action/actions/policy

    Args:
        policy_path: Path to policy_out.pkl.

    Returns:
        Actions as a numpy array of shape (T, action_dim).
    """
    with open(policy_path, "rb") as file:
        raw_actions = pickle.load(file)

    if len(raw_actions) == 0:
        raise ValueError(f"No actions found in policy file: {policy_path}")

    if isinstance(raw_actions[0], dict):
        action_key = next(
            (
                key
                for key in ["action", "actions", "policy"]
                if key in raw_actions[0]
            ),
            list(raw_actions[0].keys())[0],
        )

        clean_actions = [
            step[action_key]
            for step in raw_actions
        ]

        return np.vstack(clean_actions).astype(np.float32)

    return np.vstack(raw_actions).astype(np.float32)


def build_robot_state_features(
    obs_data: Dict[str, Any],
    state_source: str = "state",
) -> np.ndarray:
    """
    Build robot-state features from obs_dict.pkl content.

    Supported state_source values:
        - "state"
        - "qpos"
        - "qpos_qvel"
        - "state_qpos"
        - "state_qpos_qvel"

    Args:
        obs_data: Loaded obs_dict.pkl dictionary.
        state_source: Which robot state representation to use.

    Returns:
        Robot state features as a numpy array of shape (T, robot_state_dim).
    """
    if state_source == "state":
        return np.asarray(obs_data["state"], dtype=np.float32)

    if state_source == "qpos":
        return np.asarray(obs_data["qpos"], dtype=np.float32)

    if state_source == "qpos_qvel":
        qpos = np.asarray(obs_data["qpos"], dtype=np.float32)
        qvel = np.asarray(obs_data["qvel"], dtype=np.float32)
        return np.concatenate([qpos, qvel], axis=1)

    if state_source == "state_qpos":
        state = np.asarray(obs_data["state"], dtype=np.float32)
        qpos = np.asarray(obs_data["qpos"], dtype=np.float32)
        return np.concatenate([state, qpos], axis=1)

    if state_source == "state_qpos_qvel":
        state = np.asarray(obs_data["state"], dtype=np.float32)
        qpos = np.asarray(obs_data["qpos"], dtype=np.float32)
        qvel = np.asarray(obs_data["qvel"], dtype=np.float32)
        return np.concatenate([state, qpos, qvel], axis=1)

    raise ValueError(f"Unsupported state_source: {state_source}")


class MultiModalRoboticDataset(Dataset):
    """
    PyTorch Dataset for multimodal robotic behavioral cloning.

    Each sample contains:
        - A temporal window of OpenCLIP visual embeddings
        - A temporal window of robot-state features
        - A target action vector

    Recommended workflow:
        1. Run robotics_data_prep/filter_data.py to create a filtered CSV.
        2. Run openclip_embed.py to precompute embeddings.
        3. Train using this dataset with use_precomputed_embeddings=True.

    Input shape returned per sample:
        fused_features: (seq_length, input_dim)

    Target shape returned per sample:
        target_action: (action_dim,)
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        csv_path: Optional[str] = None,
        status_filter: Optional[str] = "SUCCESS",
        seq_length: int = 10,
        action_delay: int = 0,
        target_type: str = "action_at_window_end",
        state_source: str = "state",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        use_robot_state: bool = True,
        use_precomputed_embeddings: bool = True,
        compute_missing_embeddings: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_root: Root directory for raw trajectories.
            csv_path: Optional filtered CSV file with Full_Path and Status columns.
            status_filter: Optional status filter for the CSV file.
            seq_length: Number of timesteps in each input window.
            action_delay: Offset applied to the target action index.
            target_type: Target definition. Supports:
                         "action_at_window_end" and "action_next_step".
            state_source: Which robot-state features to use.
            clip_model_name: OpenCLIP model name.
            clip_pretrained: OpenCLIP pretrained weights name.
            use_robot_state: If False, only visual embeddings are used.
            use_precomputed_embeddings: If True, load .npy embeddings from trajectory folders.
            compute_missing_embeddings: If True, compute OpenCLIP embeddings when .npy is missing.
                                        If False, trajectories without embeddings are skipped.
            device: Device used for OpenCLIP embedding extraction when needed.
        """
        if seq_length < 1:
            raise ValueError("seq_length must be at least 1.")

        if action_delay < 0:
            raise ValueError("action_delay must be non-negative.")

        if target_type not in ["action_at_window_end", "action_next_step"]:
            raise ValueError(
                "target_type must be either 'action_at_window_end' "
                "or 'action_next_step'."
            )

        if data_root is None and csv_path is None:
            raise ValueError("Either data_root or csv_path must be provided.")

        self.data_root = data_root
        self.csv_path = csv_path
        self.status_filter = status_filter
        self.seq_length = seq_length
        self.action_delay = action_delay
        self.target_type = target_type
        self.state_source = state_source
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.use_robot_state = use_robot_state
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.compute_missing_embeddings = compute_missing_embeddings

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.embedding_file_name = get_embedding_file_name(
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
        )

        self.clip_model = None
        self.clip_preprocess = None

        # Load OpenCLIP only if we may need to compute embeddings online.
        if not self.use_precomputed_embeddings or self.compute_missing_embeddings:
            self._load_openclip_model()

        self.samples: List[Tuple[str, int, int]] = []
        self.traj_data: Dict[str, Dict[str, Any]] = {}

        self.num_missing_embedding_trajectories = 0
        self.num_loaded_embedding_trajectories = 0
        self.num_online_embedding_trajectories = 0

        self._initialize_trajectories()

    def _load_openclip_model(self) -> None:
        """
        Load OpenCLIP model and preprocessing pipeline.

        This is only needed when embeddings are computed online.
        """
        if self.clip_model is not None and self.clip_preprocess is not None:
            return

        print(
            f"Loading OpenCLIP model: {self.clip_model_name} "
            f"({self.clip_pretrained}) on {self.device}"
        )

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
        )

        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def _get_trajectory_folders(self) -> List[str]:
        """
        Get trajectory folders either from CSV or by recursively scanning data_root.
        """
        if self.csv_path is not None:
            return load_trajectory_paths_from_csv(
                csv_path=self.csv_path,
                status_filter=self.status_filter,
            )

        return find_valid_trajectory_folders(self.data_root)

    def _compute_target_index(self, start_idx: int) -> int:
        """
        Compute the target action index for a given window start index.
        """
        window_end_idx = start_idx + self.seq_length - 1

        if self.target_type == "action_at_window_end":
            return window_end_idx + self.action_delay

        if self.target_type == "action_next_step":
            return window_end_idx + 1 + self.action_delay

        raise ValueError(f"Unsupported target_type: {self.target_type}")

    def _load_precomputed_embeddings(
        self,
        traj_path: str,
    ) -> Optional[np.ndarray]:
        """
        Load precomputed OpenCLIP embeddings for one trajectory.

        Args:
            traj_path: Path to one trajectory folder.

        Returns:
            Embeddings array if the file exists, otherwise None.
        """
        embedding_path = os.path.join(traj_path, self.embedding_file_name)

        if not os.path.exists(embedding_path):
            return None

        embeddings = np.load(embedding_path).astype(np.float32)
        return embeddings

    def _compute_embeddings_for_trajectory(
        self,
        images_dir: str,
        image_files: List[str],
    ) -> np.ndarray:
        """
        Compute OpenCLIP embeddings for one trajectory.

        This is a fallback path. For fast training, prefer precomputed embeddings.

        Args:
            images_dir: Path to images0 folder.
            image_files: Ordered list of image filenames.

        Returns:
            Embeddings array of shape (T, visual_embedding_dim).
        """
        self._load_openclip_model()

        image_tensors = []

        for image_name in image_files:
            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.clip_preprocess(image)
            image_tensors.append(image_tensor)

        images_stacked = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            visual_features = self.clip_model.encode_image(images_stacked)
            visual_features = visual_features.float().cpu().numpy()

        return visual_features.astype(np.float32)

    def _initialize_trajectories(self) -> None:
        """
        Load trajectory metadata and build sample indices.
        """
        trajectory_folders = self._get_trajectory_folders()

        print(f"Found {len(trajectory_folders)} candidate trajectory folders.")

        skipped_count = 0

        for traj_path in trajectory_folders:
            obs_path = os.path.join(traj_path, "obs_dict.pkl")
            policy_path = os.path.join(traj_path, "policy_out.pkl")
            images_dir = os.path.join(traj_path, "images0")

            if not os.path.exists(obs_path):
                skipped_count += 1
                continue

            if not os.path.exists(policy_path):
                skipped_count += 1
                continue

            if not os.path.isdir(images_dir):
                skipped_count += 1
                continue

            try:
                with open(obs_path, "rb") as file:
                    obs_data = pickle.load(file)

                robot_features = build_robot_state_features(
                    obs_data=obs_data,
                    state_source=self.state_source,
                )

                actions = load_policy_actions(policy_path)
                image_files = list_image_files(images_dir)

                if self.use_precomputed_embeddings:
                    visual_features = self._load_precomputed_embeddings(traj_path)

                    if visual_features is None:
                        self.num_missing_embedding_trajectories += 1

                        if self.compute_missing_embeddings:
                            visual_features = self._compute_embeddings_for_trajectory(
                                images_dir=images_dir,
                                image_files=image_files,
                            )
                            self.num_online_embedding_trajectories += 1
                        else:
                            skipped_count += 1
                            continue

                    else:
                        self.num_loaded_embedding_trajectories += 1

                else:
                    visual_features = self._compute_embeddings_for_trajectory(
                        images_dir=images_dir,
                        image_files=image_files,
                    )
                    self.num_online_embedding_trajectories += 1

                num_steps = min(
                    len(robot_features),
                    len(actions),
                    len(image_files),
                    len(visual_features),
                )

                if num_steps <= self.seq_length:
                    skipped_count += 1
                    continue

                # Keep synchronized arrays only.
                robot_features = robot_features[:num_steps]
                actions = actions[:num_steps]
                image_files = image_files[:num_steps]
                visual_features = visual_features[:num_steps]

                valid_starts = []

                for start_idx in range(0, num_steps - self.seq_length + 1):
                    target_idx = self._compute_target_index(start_idx)

                    if target_idx >= num_steps:
                        continue

                    valid_starts.append((start_idx, target_idx))

                if not valid_starts:
                    skipped_count += 1
                    continue

                self.traj_data[traj_path] = {
                    "robot_features": robot_features,
                    "actions": actions,
                    "image_files": image_files,
                    "images_dir": images_dir,
                    "visual_features": visual_features,
                    "num_steps": num_steps,
                }

                for start_idx, target_idx in valid_starts:
                    self.samples.append((traj_path, start_idx, target_idx))

            except Exception as error:
                print(f"Warning: Skipping trajectory due to error: {traj_path}")
                print(f"Reason: {error}")
                skipped_count += 1
                continue

        print("Dataset initialization complete.")
        print(f"Valid trajectories loaded: {len(self.traj_data)}")
        print(f"Training windows created: {len(self.samples)}")
        print(f"Skipped trajectories: {skipped_count}")
        print(f"Embeddings loaded from .npy: {self.num_loaded_embedding_trajectories}")
        print(f"Embeddings computed online: {self.num_online_embedding_trajectories}")
        print(f"Missing embedding trajectories: {self.num_missing_embedding_trajectories}")


    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Return metadata for one dataset window without changing __getitem__ output.

        This is used by evaluation/results screens to connect a saved prediction
        example back to its trajectory and image frame.
        """
        traj_path, start_idx, target_idx = self.samples[idx]
        traj = self.traj_data[traj_path]

        image_files = traj.get("image_files", [])
        images_dir = traj.get("images_dir", "")

        target_image_name = None
        target_frame_path = None

        if 0 <= target_idx < len(image_files):
            target_image_name = image_files[target_idx]
            target_frame_path = os.path.abspath(
                os.path.join(images_dir, target_image_name)
            )

        window_image_files = image_files[start_idx : start_idx + self.seq_length]
        window_frame_paths = [
            os.path.abspath(os.path.join(images_dir, image_name))
            for image_name in window_image_files
        ]

        return {
            "dataset_index": int(idx),
            "trajectory_path": os.path.abspath(traj_path),
            "trajectory_id": os.path.basename(os.path.abspath(traj_path)),
            "images_dir": os.path.abspath(images_dir),
            "start_idx": int(start_idx),
            "target_idx": int(target_idx),
            "frame_index": int(target_idx),
            "target_image_name": target_image_name,
            "frame_path": target_frame_path,
            "target_frame_path": target_frame_path,
            "window_frame_paths": window_frame_paths,
        }

    def __len__(self) -> int:
        """
        Return the number of training samples/windows.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one training sample.

        Returns:
            fused_features: Tensor of shape (seq_length, input_dim).
            target_action: Tensor of shape (action_dim,).
        """
        traj_path, start_idx, target_idx = self.samples[idx]
        traj = self.traj_data[traj_path]

        visual_window = traj["visual_features"][
            start_idx : start_idx + self.seq_length
        ]

        visual_window_tensor = torch.tensor(visual_window, dtype=torch.float32)

        if self.use_robot_state:
            robot_window = traj["robot_features"][
                start_idx : start_idx + self.seq_length
            ]

            robot_window_tensor = torch.tensor(robot_window, dtype=torch.float32)

            fused_features = torch.cat(
                [visual_window_tensor, robot_window_tensor],
                dim=1,
            )
        else:
            fused_features = visual_window_tensor

        target_action = traj["actions"][target_idx]
        target_action_tensor = torch.tensor(target_action, dtype=torch.float32)

        return fused_features, target_action_tensor

    def get_stats(self) -> Dict[str, Any]:
        """
        Return dataset statistics for logging and metrics.json.
        """
        if len(self.traj_data) == 0:
            robot_state_dim = 0
            action_dim = 0
            visual_embedding_dim = 0
        else:
            first_traj = next(iter(self.traj_data.values()))
            robot_state_dim = int(first_traj["robot_features"].shape[1])
            action_dim = int(first_traj["actions"].shape[1])
            visual_embedding_dim = int(first_traj["visual_features"].shape[1])

        input_dim = (
            visual_embedding_dim + robot_state_dim
            if self.use_robot_state
            else visual_embedding_dim
        )

        return {
            "num_trajectories": len(self.traj_data),
            "num_windows": len(self.samples),
            "seq_length": self.seq_length,
            "action_delay": self.action_delay,
            "target_type": self.target_type,
            "state_source": self.state_source,
            "use_robot_state": self.use_robot_state,
            "use_precomputed_embeddings": self.use_precomputed_embeddings,
            "compute_missing_embeddings": self.compute_missing_embeddings,
            "robot_state_dim": robot_state_dim,
            "visual_embedding_dim": visual_embedding_dim,
            "input_dim": input_dim,
            "action_dim": action_dim,
            "clip_model_name": self.clip_model_name,
            "clip_pretrained": self.clip_pretrained,
            "embedding_file_name": self.embedding_file_name,
            "num_loaded_embedding_trajectories": self.num_loaded_embedding_trajectories,
            "num_online_embedding_trajectories": self.num_online_embedding_trajectories,
            "num_missing_embedding_trajectories": self.num_missing_embedding_trajectories,
        }


def test_dataloader() -> None:
    """
    Run a small dataloader sanity check.

    Update csv_path before running this test locally.
    """
    csv_path = "/home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv"

    dataset = MultiModalRoboticDataset(
        csv_path=csv_path,
        status_filter="SUCCESS",
        seq_length=10,
        action_delay=0,
        target_type="action_at_window_end",
        state_source="state",
        use_precomputed_embeddings=True,
        compute_missing_embeddings=False,
    )

    print("Dataset stats:")
    print(dataset.get_stats())

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    fused_inputs, actions = next(iter(dataloader))

    print("Batch shape check:")
    print(f"Fused input shape: {fused_inputs.shape}")
    print(f"Target action shape: {actions.shape}")


if __name__ == "__main__":
    test_dataloader()