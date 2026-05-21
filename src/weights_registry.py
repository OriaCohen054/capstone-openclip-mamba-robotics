import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List


# Project root is one folder above src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
WEIGHTS_REGISTRY_PATH = os.path.join(CHECKPOINTS_DIR, "weights_registry.json")


# These keys define whether two training runs use the same meaningful setup.
# Runtime outputs such as best_loss, created_at, and checkpoint paths should NOT be included here.
CONFIG_HASH_KEYS = [
    "dataset_name",
    "seq_length",
    "action_delay",
    "target_type",

    "input_dim",
    "visual_embedding_dim",
    "robot_state_dim",
    "feature_fusion",
    "state_source",
    "use_robot_state",

    "clip_model_name",
    "clip_pretrained",
    "freeze_clip",

    "d_model",
    "d_state",
    "d_conv",
    "expand",
    "num_layers",
    "dropout",
    "action_dim",

    "batch_size",
    "learning_rate",
    "max_epochs",
    "patience",
    "optimizer",
    "loss_function",
    "weight_decay",
    "gradient_clip",
    "mixed_precision",

    "train_split",
    "split_strategy",
    "random_seed",

    "use_yolo_filter",
    "status_filter",
    "filter_task",
    "yolo_filter_threshold",
]


def ensure_weights_registry_exists() -> None:
    """
    Create the checkpoints directory and the weights registry file if needed.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_REGISTRY_PATH):
        default_registry = {
            "version": 1,
            "weights": []
        }

        with open(WEIGHTS_REGISTRY_PATH, "w", encoding="utf-8") as file:
            json.dump(default_registry, file, indent=4)


def load_weights_registry() -> Dict[str, Any]:
    """
    Load the weights registry JSON file.

    Returns:
        A dictionary that contains a "weights" list.
    """
    ensure_weights_registry_exists()

    try:
        with open(WEIGHTS_REGISTRY_PATH, "r", encoding="utf-8") as file:
            registry = json.load(file)

        # Backward compatibility for old registry files.
        if "weights" not in registry:
            registry["weights"] = []

        if "version" not in registry:
            registry["version"] = 1

        return registry

    except Exception:
        return {
            "version": 1,
            "weights": []
        }


def save_weights_registry(registry: Dict[str, Any]) -> None:
    """
    Save the weights registry JSON file.
    """
    ensure_weights_registry_exists()

    with open(WEIGHTS_REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


def normalize_config_value(value: Any) -> Any:
    """
    Normalize config values before hashing.

    This prevents small formatting differences from creating different hashes.
    """
    if isinstance(value, float):
        return round(value, 10)

    if isinstance(value, (int, str, bool)) or value is None:
        return value

    return str(value)


def build_training_config(
    *,
    dataset_name: str = "unknown_dataset",
    seq_length: int = 10,
    action_delay: int = 0,
    target_type: str = "action_at_window_end",

    input_dim: int = 519,
    visual_embedding_dim: int = 512,
    robot_state_dim: int = 7,
    feature_fusion: str = "openclip_plus_state_concat",
    state_source: str = "state",
    use_robot_state: bool = True,

    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    freeze_clip: bool = True,

    d_model: int = 128,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    num_layers: int = 1,
    dropout: float = 0.0,
    action_dim: int = 7,

    batch_size: int = 16,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    patience: int = 5,
    optimizer: str = "Adam",
    loss_function: str = "MSELoss",
    weight_decay: float = 0.0,
    gradient_clip: Optional[float] = None,
    mixed_precision: str = "none",

    train_split: float = 0.8,
    split_strategy: str = "random_window_split",
    random_seed: int = 42,

    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a full training configuration dictionary.

    This config is used both for documentation and for checkpoint matching.
    """
    config = {
        "dataset_name": dataset_name,
        "seq_length": int(seq_length),
        "action_delay": int(action_delay),
        "target_type": target_type,

        "input_dim": int(input_dim),
        "visual_embedding_dim": int(visual_embedding_dim),
        "robot_state_dim": int(robot_state_dim),
        "feature_fusion": feature_fusion,
        "state_source": state_source,
        "use_robot_state": bool(use_robot_state),

        "clip_model_name": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "freeze_clip": bool(freeze_clip),

        "d_model": int(d_model),
        "d_state": int(d_state),
        "d_conv": int(d_conv),
        "expand": int(expand),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "action_dim": int(action_dim),

        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "optimizer": optimizer,
        "loss_function": loss_function,
        "weight_decay": float(weight_decay),
        "gradient_clip": gradient_clip,
        "mixed_precision": mixed_precision,

        "train_split": float(train_split),
        "split_strategy": split_strategy,
        "random_seed": int(random_seed),
    }

    # Allow future parameters without breaking the function signature.
    if extra_config:
        config.update(extra_config)

    return {
        key: normalize_config_value(value)
        for key, value in config.items()
    }


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a stable short hash from the meaningful training configuration.

    Only CONFIG_HASH_KEYS are used. This prevents paths, timestamps, and metrics
    from changing the identity of the model setup.
    """
    relevant_config = {
        key: normalize_config_value(config.get(key))
        for key in CONFIG_HASH_KEYS
    }

    config_json = json.dumps(
        relevant_config,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]


def resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    """
    Convert a relative project path to an absolute path.

    Args:
        path_value: Relative or absolute path.

    Returns:
        Absolute path, or None if no path was provided.
    """
    if not path_value:
        return None

    if os.path.isabs(path_value):
        return path_value

    return os.path.join(PROJECT_ROOT, path_value)


def find_existing_weights_by_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find an existing checkpoint with the same config hash.

    Returns:
        The registry item if a matching checkpoint exists.
        Otherwise, None.
    """
    registry = load_weights_registry()
    target_hash = compute_config_hash(config)

    for item in registry.get("weights", []):
        if item.get("config_hash") != target_hash:
            continue

        weights_path = item.get("weights_path")
        absolute_weights_path = resolve_project_path(weights_path)

        if absolute_weights_path and os.path.exists(absolute_weights_path):
            return item

    return None


def register_trained_weights(
    *,
    run_name: str,
    weights_path: str,
    config: Dict[str, Any],
    best_loss: Optional[float] = None,
    config_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    status: str = "available",
    overwrite_existing: bool = False,
) -> Dict[str, Any]:
    """
    Register a trained checkpoint in weights_registry.json.

    Args:
        run_name: Human-readable run name.
        weights_path: Path to the saved model weights file.
        config: Training configuration dictionary.
        best_loss: Best validation loss, if available.
        config_path: Optional path to a saved config.json file.
        metrics_path: Optional path to a saved metrics.json file.
        status: Registry status, for example "available" or "archived".
        overwrite_existing: If True, replace an existing item with the same config hash.

    Returns:
        The registry item that was added.
    """
    registry = load_weights_registry()
    config_hash = compute_config_hash(config)

    new_item = {
        "run_name": run_name,
        "weights_path": weights_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": config_hash,
        "best_loss": best_loss,
        "status": status,
        "config": config,
    }

    if overwrite_existing:
        registry["weights"] = [
            item for item in registry.get("weights", [])
            if item.get("config_hash") != config_hash
        ]

    registry["weights"].append(new_item)
    save_weights_registry(registry)

    return new_item


def save_run_config(config_path: str, config: Dict[str, Any]) -> None:
    """
    Save a training configuration to config.json.
    """
    absolute_config_path = resolve_project_path(config_path)

    if absolute_config_path is None:
        raise ValueError("config_path cannot be empty.")

    os.makedirs(os.path.dirname(absolute_config_path), exist_ok=True)

    with open(absolute_config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)


def save_run_metrics(metrics_path: str, metrics: Dict[str, Any]) -> None:
    """
    Save training metrics to metrics.json.
    """
    absolute_metrics_path = resolve_project_path(metrics_path)

    if absolute_metrics_path is None:
        raise ValueError("metrics_path cannot be empty.")

    os.makedirs(os.path.dirname(absolute_metrics_path), exist_ok=True)

    with open(absolute_metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def format_weight_option(weight_item: Dict[str, Any]) -> str:
    """
    Create a readable label for a Streamlit selectbox.
    """
    run_name = weight_item.get("run_name", "unnamed_run")
    config = weight_item.get("config", {})

    seq_length = config.get("seq_length", weight_item.get("seq_length", "?"))
    d_model = config.get("d_model", weight_item.get("d_model", "?"))
    learning_rate = config.get("learning_rate", weight_item.get("learning_rate", "?"))
    batch_size = config.get("batch_size", weight_item.get("batch_size", "?"))
    action_delay = config.get("action_delay", "?")
    best_loss = weight_item.get("best_loss", None)

    if best_loss is None:
        loss_text = "loss=?"
    else:
        try:
            loss_text = f"loss={float(best_loss):.4f}"
        except Exception:
            loss_text = f"loss={best_loss}"

    return (
        f"{run_name} | "
        f"K={seq_length} | "
        f"d={d_model} | "
        f"lr={learning_rate} | "
        f"batch={batch_size} | "
        f"delay={action_delay} | "
        f"{loss_text}"
    )


def get_weight_options() -> Dict[str, Dict[str, Any]]:
    """
    Return display labels mapped to registry items.

    This is useful for Streamlit selectboxes.
    """
    registry = load_weights_registry()
    weights = registry.get("weights", [])

    options = {}

    for item in weights:
        if item.get("status") == "archived":
            continue

        label = format_weight_option(item)
        options[label] = item

    return options


def list_registered_weights(include_archived: bool = False) -> List[Dict[str, Any]]:
    """
    Return all registered weights.

    Args:
        include_archived: If True, include archived models too.
    """
    registry = load_weights_registry()
    weights = registry.get("weights", [])

    if include_archived:
        return weights

    return [
        item for item in weights
        if item.get("status") != "archived"
    ]


def get_weights_path(weight_item: Dict[str, Any]) -> Optional[str]:
    """
    Return the absolute path to a registered weights file.
    """
    return resolve_project_path(weight_item.get("weights_path"))


def mark_weight_as_archived(config_hash: str) -> bool:
    """
    Mark a registered checkpoint as archived without deleting the file.

    Args:
        config_hash: The config hash of the model to archive.

    Returns:
        True if an item was updated, False otherwise.
    """
    registry = load_weights_registry()
    updated = False

    for item in registry.get("weights", []):
        if item.get("config_hash") == config_hash:
            item["status"] = "archived"
            updated = True

    if updated:
        save_weights_registry(registry)

    return updated