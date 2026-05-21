# Capstone OpenCLIP + Mamba Robotics

This project implements a robotic imitation learning pipeline using OpenCLIP visual embeddings and a Mamba-based sequence model.

The goal is to learn robotic action prediction from visual trajectory sequences and robot-state data.

The project focuses on learning from successful robotic pick/lift trajectories and training a Mamba-based behavioral cloning model to predict robot actions from temporal multimodal inputs.

---

## Project Pipeline

The current pipeline is:

1. Filter raw robotic trajectories using gripper-state signals and YOLO-based visual verification.
2. Keep successful pick/lift trajectories in a filtered CSV file.
3. Precompute OpenCLIP image embeddings for successful trajectories.
4. Train a Mamba behavioral cloning model using:
   - OpenCLIP visual embeddings
   - Robot state features
   - Temporal windows
5. Save each training run with:
   - model weights
   - training configuration
   - evaluation metrics
6. Evaluate trained models using the saved checkpoint and config.

---

## Repository Structure

```text
RoboMamba_OpenCLIP/
├── src/
│   ├── robotics_data_prep/
│   │   └── filter_data.py
│   ├── openclip_embed.py
│   ├── data_loader.py
│   ├── mamba_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── trajectory_scanner.py
│   └── weights_registry.py
│
├── checkpoints/
│   ├── archive/
│   └── weights_registry.json
│
├── outputs/
│   └── logs/
│
├── docs/
├── ui/
├── data/
├── .gitignore
└── README.md
```

---

## Main Components

### `src/robotics_data_prep/filter_data.py`

Filters raw robotic trajectories using:

- gripper motor/state values
- end-effector height behavior
- YOLO-based visual verification

The output is a CSV file containing trajectory status:

```text
SUCCESS / FAILURE
```

Important note:  
The current filtering focuses on successful pick/lift behavior. It does not fully validate the complete place phase.

---

### `src/openclip_embed.py`

Precomputes OpenCLIP visual embeddings for trajectory images.

Instead of computing OpenCLIP embeddings during every training epoch, this script computes them once and saves them inside each trajectory folder as `.npy` files.

This significantly speeds up training.

---

### `src/data_loader.py`

Loads the training dataset.

It supports:

- loading trajectories from a filtered CSV
- using only rows with `Status = SUCCESS`
- loading precomputed OpenCLIP embeddings from `.npy`
- combining OpenCLIP embeddings with robot state vectors
- creating temporal windows for Mamba training

Current input shape:

```text
[seq_length, 519]
```

Where:

```text
519 = 512 OpenCLIP embedding + 7 robot state features
```

Current target shape:

```text
[7]
```

---

### `src/mamba_model.py`

Defines the Mamba-based behavioral cloning model.

The model receives temporal multimodal inputs and predicts a robot action vector.

Configurable model parameters include:

- `d_model`
- `d_state`
- `d_conv`
- `expand`
- `num_layers`
- `dropout`
- `action_dim`

---

### `src/train.py`

Trains the Mamba behavioral cloning model.

The training script supports:

- training from a filtered CSV
- optionally running YOLO filtering before training
- loading precomputed OpenCLIP embeddings
- saving checkpoints
- saving config files
- saving metrics
- updating the weights registry

Each training run is saved under:

```text
checkpoints/<run_name>/
├── model.pth
├── config.json
└── metrics.json
```

---

### `src/evaluate.py`

Evaluates a trained model.

It loads:

```text
checkpoints/<run_name>/model.pth
checkpoints/<run_name>/config.json
```

Then it rebuilds the model according to the saved config and evaluates it on the validation split or the full dataset.

---

### `src/weights_registry.py`

Manages trained checkpoints.

It stores metadata about trained models in:

```text
checkpoints/weights_registry.json
```

The registry is used to:

- list available trained models
- detect if a model with the same hyperparameters already exists
- avoid unnecessary retraining
- connect the UI to saved checkpoints

---

### `src/trajectory_scanner.py`

Scans dataset folders and detects structurally valid trajectory folders.

A valid trajectory folder contains:

```text
obs_dict.pkl
policy_out.pkl
images0/
```

This file only checks folder structure.  
It does not decide whether the robot succeeded in the task.

---

## Step 1: Filter Trajectories

Run YOLO + gripper-based filtering:

```bash
python src/robotics_data_prep/filter_data.py \
  --data_dir /home/linuxu/Downloads/scripted_6_18/scripted_raw \
  --weights src/robotics_data_prep/best.pt \
  --output final_mamba_dataset.csv
```

Expected output:

```text
/home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv
```

The CSV contains:

```text
Trajectory
Full_Path
Status
Stable_Motor_Value
Reason
```

Only rows with:

```text
Status = SUCCESS
```

are used for training.

---

## Step 2: Precompute OpenCLIP Embeddings

Run:

```bash
python src/openclip_embed.py \
  --csv_path /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32
```

To recompute and overwrite existing embeddings:

```bash
python src/openclip_embed.py \
  --csv_path /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32 \
  --overwrite
```

Expected result:

```text
Created: 4172
Skipped existing: 0
Failed: 0
```

After this step, each successful trajectory folder should contain a file similar to:

```text
openclip_vitb32_laion2b_s34b_b79k_embeddings.npy
```

---

## Step 3: Train Mamba Model

Example sanity-check training run:

```bash
python src/train.py \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --seq_length 10 \
  --d_model 128 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --max_epochs 1 \
  --patience 1 \
  --run_name test_pipeline_precomputed \
  --force_train
```

Expected saved files:

```text
checkpoints/test_pipeline_precomputed/model.pth
checkpoints/test_pipeline_precomputed/config.json
checkpoints/test_pipeline_precomputed/metrics.json
outputs/logs/test_pipeline_precomputed.txt
```

Example output from the verified test run:

```text
Valid trajectories loaded: 4172
Training windows created: 166718
Embeddings loaded from .npy: 4172
Embeddings computed online: 0
Missing embedding trajectories: 0

Epoch 1/1 | Train Loss: 0.010972 | Val Loss: 0.007486 | new_best

Best validation loss: 0.007486
```

---

## Step 4: Evaluate Trained Model

Evaluate a trained run:

```bash
python src/evaluate.py \
  --run_name test_pipeline_precomputed \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32 \
  --show_examples 3
```

Quick evaluation on only a few batches:

```bash
python src/evaluate.py \
  --run_name test_pipeline_precomputed \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32 \
  --max_batches 10 \
  --show_examples 3
```

Example verified evaluation result after one epoch:

```text
Evaluated samples: 33344
MSE: 0.007486
MAE: 0.029676
Per-dimension MAE: [0.0074 0.0102 0.0134 0.0097 0.008  0.057  0.102 ]
```

This result is only a pipeline sanity check, not a final trained model.

---

## Current Verified Status

The full code pipeline was tested successfully with:

```text
4172 successful trajectories
166718 training windows
Input shape: torch.Size([10, 519])
Target shape: torch.Size([7])
```

Precomputed embeddings were loaded successfully:

```text
Embeddings loaded from .npy: 4172
Embeddings computed online: 0
Missing embedding trajectories: 0
```

Training completed successfully.

Evaluation completed successfully.

---

## Suggested Full Training Run

After the pipeline sanity check passes, a more serious training run can be executed:

```bash
python src/train.py \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --seq_length 10 \
  --d_model 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --max_epochs 20 \
  --patience 5 \
  --run_name mamba_k10_d128_lr0001_pick_success_v1 \
  --force_train
```

Possible future experiments:

```text
seq_length = 20
d_model = 256
batch_size = 32
learning_rate = 0.0005
```

---

## Important Notes

### Pick/Lift Filtering

The current YOLO + gripper filtering stage focuses on successful pick/lift behavior.

It should not be described as complete full-task success validation because not every trajectory is fully checked for the place phase.

---

### Precomputed Embeddings

Training should use precomputed OpenCLIP embeddings when available.

This avoids recomputing OpenCLIP for overlapping windows and makes training significantly faster.

---

### Checkpoints

Training outputs are saved under:

```text
checkpoints/<run_name>/
```

Each run contains:

```text
model.pth
config.json
metrics.json
```

Old or experimental checkpoints should be moved to:

```text
checkpoints/archive/
```

---

### Logs

Training logs are saved under:

```text
outputs/logs/
```

---

### Git Ignore

Large files should not be committed to Git.

Recommended ignored files/folders:

```gitignore
__pycache__/
*.pyc

data/
datasets/

checkpoints/
outputs/
logs/

*.pth
*.pt
*.ckpt

runs/
wandb/
```

If small sample data is added later, it can be kept under:

```text
data/sample/
```

with a custom `.gitignore` exception.

---

## Known Warning

During training, PyTorch may show:

```text
FutureWarning: torch.cuda.amp.GradScaler is deprecated
```

This is not an error.  
The current training pipeline works correctly.

This can be updated later to the newer `torch.amp.GradScaler` API, but it is not urgent.

---

## Current Default Configuration

The current verified configuration is:

```json
{
  "dataset_name": "final_mamba_dataset.csv",
  "seq_length": 10,
  "action_delay": 0,
  "target_type": "action_at_window_end",

  "input_dim": 519,
  "visual_embedding_dim": 512,
  "robot_state_dim": 7,
  "feature_fusion": "openclip_plus_state_concat",
  "state_source": "state",
  "use_robot_state": true,

  "clip_model_name": "ViT-B-32",
  "clip_pretrained": "laion2b_s34b_b79k",
  "freeze_clip": true,

  "d_model": 128,
  "d_state": 16,
  "d_conv": 4,
  "expand": 2,
  "num_layers": 1,
  "dropout": 0.0,
  "action_dim": 7,

  "batch_size": 16,
  "learning_rate": 0.001,
  "max_epochs": 20,
  "patience": 5,
  "optimizer": "Adam",
  "loss_function": "MSELoss",
  "weight_decay": 0.0,
  "gradient_clip": null,
  "mixed_precision": "none",

  "train_split": 0.8,
  "split_strategy": "random_window_split",
  "random_seed": 42
}
```

---

## Project Status

The project currently has a working end-to-end pipeline:

```text
Raw trajectories
→ YOLO/gripper filtering
→ SUCCESS CSV
→ OpenCLIP precomputed embeddings
→ Mamba behavioral cloning training
→ checkpoint/config/metrics saving
→ evaluation
```

Next planned steps:

1. Run a longer training experiment.
2. Update the UI to use `weights_registry.json`.
3. Improve README with final training results.
4. Optionally add sample data for reproducibility.
