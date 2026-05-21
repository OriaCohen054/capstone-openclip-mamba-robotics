import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_loader import MultiModalRoboticDataset
from mamba_model import MultimodalMambaBC
from weights_registry import PROJECT_ROOT, load_weights_registry, get_weights_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Mamba behavioral cloning model."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of a run folder under checkpoints/.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a checkpoint directory containing model.pth and config.json.",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Direct path to a model checkpoint file.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Direct path to a config.json file.",
    )

    parser.add_argument(
        "--filtered_csv",
        type=str,
        default=None,
        help="Path to the filtered CSV file used for evaluation.",
    )

    parser.add_argument(
        "--status_filter",
        type=str,
        default="SUCCESS",
        help="Only trajectories with this Status value will be evaluated.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional limit on number of batches to evaluate.",
    )

    parser.add_argument(
        "--show_examples",
        type=int,
        default=3,
        help="Number of prediction examples to print.",
    )

    parser.add_argument(
        "--evaluate_split",
        type=str,
        default="validation",
        choices=["validation", "all"],
        help="Evaluate on validation split or on the full dataset.",
    )

    return parser.parse_args()


def project_path(path_value: str) -> str:
    """
    Convert a relative project path to an absolute path.
    """
    if os.path.isabs(path_value):
        return path_value

    return os.path.join(PROJECT_ROOT, path_value)


def load_json_file(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_checkpoint_paths(args: argparse.Namespace) -> Dict[str, str]:
    """
    Resolve model and config paths from run_name, checkpoint_dir, or direct paths.

    Returns:
        Dictionary with weights_path and config_path.
    """
    if args.run_name:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.run_name)
        weights_path = os.path.join(checkpoint_dir, "model.pth")
        config_path = os.path.join(checkpoint_dir, "config.json")

        return {
            "weights_path": weights_path,
            "config_path": config_path,
        }

    if args.checkpoint_dir:
        checkpoint_dir = project_path(args.checkpoint_dir)
        weights_path = os.path.join(checkpoint_dir, "model.pth")
        config_path = os.path.join(checkpoint_dir, "config.json")

        return {
            "weights_path": weights_path,
            "config_path": config_path,
        }

    if args.weights_path and args.config_path:
        return {
            "weights_path": project_path(args.weights_path),
            "config_path": project_path(args.config_path),
        }

    raise ValueError(
        "You must provide one of the following:\n"
        "1. --run_name\n"
        "2. --checkpoint_dir\n"
        "3. both --weights_path and --config_path"
    )


def load_checkpoint(weights_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint.

    The training script saves a dictionary with model_state_dict and config.
    This function also supports older checkpoints that contain only a state_dict.
    """
    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint

    # Backward compatibility for older .pth files that directly stored state_dict.
    return {
        "model_state_dict": checkpoint,
        "config": None,
        "best_val_loss": None,
        "best_epoch": None,
    }


def build_model_from_config(config: Dict[str, Any], device: torch.device) -> MultimodalMambaBC:
    """
    Build the Mamba model according to the saved config.
    """
    model = MultimodalMambaBC(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        action_dim=int(config["action_dim"]),
        d_state=int(config.get("d_state", 16)),
        d_conv=int(config.get("d_conv", 4)),
        expand=int(config.get("expand", 2)),
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
    )

    return model.to(device)


def build_evaluation_dataset(
    config: Dict[str, Any],
    filtered_csv: Optional[str],
    status_filter: str,
) -> MultiModalRoboticDataset:
    """
    Build the dataset used for evaluation.

    Args:
        config: Saved training configuration.
        filtered_csv: Optional CSV path. If not provided, user must pass it.
        status_filter: Status value used for filtering rows from CSV.

    Returns:
        MultiModalRoboticDataset instance.
    """
    if not filtered_csv:
        raise ValueError(
            "Please provide --filtered_csv for evaluation.\n"
            "Example:\n"
            "python src/evaluate.py --run_name test_pipeline_precomputed "
            "--filtered_csv /path/to/final_mamba_dataset.csv"
        )

    return MultiModalRoboticDataset(
        csv_path=filtered_csv,
        status_filter=status_filter,
        seq_length=int(config["seq_length"]),
        action_delay=int(config.get("action_delay", 0)),
        target_type=str(config.get("target_type", "action_at_window_end")),
        state_source=str(config.get("state_source", "state")),
        clip_model_name=str(config.get("clip_model_name", "ViT-B-32")),
        clip_pretrained=str(config.get("clip_pretrained", "laion2b_s34b_b79k")),
        use_robot_state=bool(config.get("use_robot_state", True)),
        use_precomputed_embeddings=True,
        compute_missing_embeddings=False,
    )


def build_eval_loader(
    dataset: MultiModalRoboticDataset,
    config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    evaluate_split: str,
) -> DataLoader:
    """
    Build an evaluation DataLoader.

    If evaluate_split is 'validation', reproduce the same random window split
    used by train.py and evaluate only on the validation part.
    """
    if evaluate_split == "all":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    train_split = float(config.get("train_split", 0.8))
    random_seed = int(config.get("random_seed", 42))

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid train/validation split for evaluation.")

    split_generator = torch.Generator().manual_seed(random_seed)

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def evaluate_model(
    model: MultimodalMambaBC,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    show_examples: int,
) -> Dict[str, Any]:
    """
    Run model evaluation and compute regression metrics.

    Metrics:
        - MSE
        - MAE
        - Per-dimension MAE
    """
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    per_dim_abs_error_sum = None
    printed_examples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            squared_error = (predictions - targets) ** 2
            absolute_error = torch.abs(predictions - targets)

            batch_size = targets.shape[0]

            total_mse += squared_error.mean(dim=1).sum().item()
            total_mae += absolute_error.mean(dim=1).sum().item()
            total_samples += batch_size

            per_dim_batch_sum = absolute_error.sum(dim=0).detach().cpu().numpy()

            if per_dim_abs_error_sum is None:
                per_dim_abs_error_sum = per_dim_batch_sum
            else:
                per_dim_abs_error_sum += per_dim_batch_sum

            if printed_examples < show_examples:
                predictions_np = predictions.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()

                for example_idx in range(batch_size):
                    if printed_examples >= show_examples:
                        break

                    np.set_printoptions(precision=4, suppress=True)

                    print(f"\nExample {printed_examples + 1}:")
                    print(f"Ground truth action: {targets_np[example_idx]}")
                    print(f"Predicted action:    {predictions_np[example_idx]}")
                    print(
                        "MAE:                 "
                        f"{np.mean(np.abs(targets_np[example_idx] - predictions_np[example_idx])):.6f}"
                    )

                    printed_examples += 1

    if total_samples == 0:
        raise ValueError("No samples were evaluated.")

    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    per_dim_mae = (
        per_dim_abs_error_sum / total_samples
        if per_dim_abs_error_sum is not None
        else None
    )

    return {
        "num_samples": total_samples,
        "mse": float(avg_mse),
        "mae": float(avg_mae),
        "per_dim_mae": per_dim_mae.tolist() if per_dim_mae is not None else None,
    }


def main() -> None:
    """
    Main evaluation entry point.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    paths = resolve_checkpoint_paths(args)

    weights_path = paths["weights_path"]
    config_path = paths["config_path"]

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file was not found: {weights_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file was not found: {config_path}")

    print(f"Loading config from: {config_path}")
    config = load_json_file(config_path)

    print(f"Loading checkpoint from: {weights_path}")
    checkpoint = load_checkpoint(weights_path, device=device)

    model = build_model_from_config(config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = build_evaluation_dataset(
        config=config,
        filtered_csv=args.filtered_csv,
        status_filter=args.status_filter,
    )

    print("Dataset stats:")
    print(dataset.get_stats())

    dataloader = build_eval_loader(
        dataset=dataset,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        evaluate_split=args.evaluate_split,
    )

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
        show_examples=args.show_examples,
    )

    print("\nEvaluation complete.")
    print(f"Evaluated samples: {results['num_samples']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")

    if results["per_dim_mae"] is not None:
        print(f"Per-dimension MAE: {np.array(results['per_dim_mae'])}")


if __name__ == "__main__":
    main()