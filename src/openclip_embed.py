import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip


def natural_sort_key(text: str) -> List[Any]:
    """
    Create a natural sorting key for filenames.

    This prevents ordering issues such as:
        im_10.jpg appearing before im_2.jpg
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def list_image_files(images_dir: str) -> List[str]:
    """
    List image files from an images directory.

    Args:
        images_dir: Path to the trajectory images directory.

    Returns:
        A naturally sorted list of image filenames.
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


def find_valid_trajectory_folders(data_root: str) -> List[str]:
    """
    Recursively find valid trajectory folders.

    A valid trajectory folder must contain:
        - obs_dict.pkl
        - policy_out.pkl
        - images0/

    Args:
        data_root: Root directory containing raw trajectories.

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

    The CSV is expected to contain:
        - Full_Path
        - Status

    Args:
        csv_path: Path to the filtered CSV file.
        status_filter: If provided, only rows with this Status value are used.

    Returns:
        A sorted list of absolute trajectory folder paths.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    trajectory_paths = []

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        if "Full_Path" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'Full_Path' column.")

        for row in reader:
            full_path = str(row.get("Full_Path", "")).strip()
            status = str(row.get("Status", "")).strip()

            if not full_path:
                continue

            if status_filter is not None and status != status_filter:
                continue

            trajectory_paths.append(os.path.abspath(full_path))

    trajectory_paths = sorted(list(set(trajectory_paths)))
    return trajectory_paths


def get_embedding_file_name(
    clip_model_name: str,
    clip_pretrained: str,
) -> str:
    """
    Build a stable embedding filename from the OpenCLIP configuration.

    Args:
        clip_model_name: OpenCLIP model name.
        clip_pretrained: OpenCLIP pretrained weights name.

    Returns:
        Filename for the saved embeddings.
    """
    safe_model = clip_model_name.replace("/", "_").replace("-", "").lower()
    safe_pretrained = clip_pretrained.replace("/", "_").replace("-", "_").lower()

    return f"openclip_{safe_model}_{safe_pretrained}_embeddings.npy"


def get_metadata_file_name(embedding_file_name: str) -> str:
    """
    Build metadata filename matching an embedding file.

    Args:
        embedding_file_name: Embedding .npy filename.

    Returns:
        Metadata .json filename.
    """
    base_name = os.path.splitext(embedding_file_name)[0]
    return f"{base_name}_metadata.json"


def load_openclip_model(
    clip_model_name: str,
    clip_pretrained: str,
    device: torch.device,
):
    """
    Load OpenCLIP model and preprocessing pipeline.

    Args:
        clip_model_name: OpenCLIP model architecture name.
        clip_pretrained: OpenCLIP pretrained weights name.
        device: Torch device.

    Returns:
        Tuple of model and preprocess function.
    """
    print(
        f"Loading OpenCLIP model: {clip_model_name} "
        f"({clip_pretrained}) on {device}"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )

    model = model.to(device)
    model.eval()

    return model, preprocess


def extract_embeddings_for_images(
    image_paths: List[str],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract OpenCLIP embeddings for a list of images.

    Args:
        image_paths: List of image paths.
        model: Loaded OpenCLIP model.
        preprocess: OpenCLIP preprocessing function.
        device: Torch device.
        batch_size: Number of images per OpenCLIP forward pass.

    Returns:
        Numpy array with shape (num_images, embedding_dim).
    """
    all_embeddings = []

    for start_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        image_tensors = []

        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
            image_tensors.append(image_tensor)

        images_batch = torch.stack(image_tensors).to(device)

        with torch.no_grad():
            embeddings = model.encode_image(images_batch)
            embeddings = embeddings.float().cpu().numpy()

        all_embeddings.append(embeddings)

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(all_embeddings).astype(np.float32)


def save_trajectory_embeddings(
    traj_path: str,
    model,
    preprocess,
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    batch_size: int,
    overwrite: bool,
) -> Dict[str, Any]:
    """
    Extract and save OpenCLIP embeddings for one trajectory.

    Embeddings are saved inside the trajectory folder.

    Args:
        traj_path: Path to one trajectory folder.
        model: Loaded OpenCLIP model.
        preprocess: OpenCLIP preprocessing function.
        device: Torch device.
        clip_model_name: OpenCLIP model name.
        clip_pretrained: OpenCLIP pretrained weights name.
        batch_size: Batch size for embedding extraction.
        overwrite: Whether to overwrite existing embedding files.

    Returns:
        Dictionary with extraction metadata.
    """
    images_dir = os.path.join(traj_path, "images0")
    image_files = list_image_files(images_dir)

    embedding_file_name = get_embedding_file_name(
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    metadata_file_name = get_metadata_file_name(embedding_file_name)

    embedding_path = os.path.join(traj_path, embedding_file_name)
    metadata_path = os.path.join(traj_path, metadata_file_name)

    if os.path.exists(embedding_path) and not overwrite:
        return {
            "trajectory_path": traj_path,
            "status": "skipped_existing",
            "embedding_path": embedding_path,
            "metadata_path": metadata_path,
            "num_images": len(image_files),
        }

    image_paths = [
        os.path.join(images_dir, image_name)
        for image_name in image_files
    ]

    embeddings = extract_embeddings_for_images(
        image_paths=image_paths,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=batch_size,
    )

    np.save(embedding_path, embeddings)

    metadata = {
        "trajectory_path": traj_path,
        "embedding_path": embedding_path,
        "clip_model_name": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "num_images": len(image_files),
        "embedding_shape": list(embeddings.shape),
        "image_files": image_files,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return {
        "trajectory_path": traj_path,
        "status": "created",
        "embedding_path": embedding_path,
        "metadata_path": metadata_path,
        "num_images": len(image_files),
        "embedding_shape": list(embeddings.shape),
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Precompute OpenCLIP embeddings for robotic trajectories."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing raw trajectory folders.",
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional filtered CSV file containing Full_Path and Status columns.",
    )

    parser.add_argument(
        "--status_filter",
        type=str,
        default="SUCCESS",
        help="Only trajectories with this Status value are processed when using CSV.",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model architecture name.",
    )

    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights name.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for OpenCLIP embedding extraction.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Example: cuda, cpu. Default uses cuda if available.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for OpenCLIP embedding extraction.
    """
    args = parse_args()

    if args.csv_path is None and args.data_dir is None:
        raise ValueError("Either --csv_path or --data_dir must be provided.")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.csv_path is not None:
        trajectory_paths = load_trajectory_paths_from_csv(
            csv_path=args.csv_path,
            status_filter=args.status_filter,
        )
    else:
        trajectory_paths = find_valid_trajectory_folders(args.data_dir)

    print(f"Found {len(trajectory_paths)} trajectories for embedding extraction.")

    model, preprocess = load_openclip_model(
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )

    created_count = 0
    skipped_count = 0
    failed_count = 0

    for traj_path in tqdm(trajectory_paths, desc="Extracting OpenCLIP embeddings"):
        try:
            result = save_trajectory_embeddings(
                traj_path=traj_path,
                model=model,
                preprocess=preprocess,
                device=device,
                clip_model_name=args.clip_model_name,
                clip_pretrained=args.clip_pretrained,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )

            if result["status"] == "created":
                created_count += 1
            elif result["status"] == "skipped_existing":
                skipped_count += 1

        except Exception as error:
            failed_count += 1
            print(f"Warning: Failed to process trajectory: {traj_path}")
            print(f"Reason: {error}")

    print("\nEmbedding extraction complete.")
    print(f"Created: {created_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()