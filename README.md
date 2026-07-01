# RoboMamba: Multimodal Behavioral Cloning for a Robotic Arm Using Mamba and OpenCLIP

**Student:** Oria Cohen  
**Advisor:** Dr. Sharon Yalov-Handzel  
**Project:** Computer Science Final Project

---

## 1. Project Overview

This project implements a multimodal robotic imitation-learning pipeline for predicting the next robotic action from visual trajectory sequences and robot-state data.

The system combines:

- **YOLO-based trajectory filtering** for selecting reliable robotic demonstrations.
- **OpenCLIP visual embeddings** for converting image frames into compact 512-dimensional visual feature vectors.
- **Robot state vectors** for adding non-visual robotic information.
- **Mamba sequence modeling** for learning temporal dependencies across consecutive timesteps.
- **Behavioral Cloning** for predicting the next continuous robot action vector.
- **Streamlit UI** for operating the full pipeline end-to-end.

The final model receives a fixed-length sequence of multimodal features and predicts the next 7-dimensional robotic action.

---

## 2. Project Poster

The poster below summarizes the project motivation, solution, architecture, and main engineering challenges.

[View full poster as PDF](docs/poster/robomamba_poster.pdf)

![RoboMamba Project Poster](docs/poster/robomamba_poster.png)

---

## 3. High-Level Pipeline

```text
Raw robotic trajectories
        ↓
Dataset selection and validation
        ↓
YOLO-based trajectory filtering
        ↓
Filtered CSV: final_mamba_dataset.csv
        ↓
OpenCLIP embedding generation
        ↓
Multimodal data loading:
512D visual embedding + 7D robot state = 519D input vector
        ↓
Fixed-length temporal windows
        ↓
Mamba behavioral cloning training
        ↓
Checkpoint + config + metrics
        ↓
Evaluation
        ↓
Streamlit Results visualization
```

Current verified input and target format:

```text
Input shape per sample:  [seq_length, 519]
519 = 512 OpenCLIP embedding + 7 robot-state features

Target shape per sample: [7]
Target = next robotic action vector
```

---

## 4. Main Technologies

- Python
- PyTorch
- CUDA
- Streamlit
- OpenCLIP
- Mamba-SSM
- Ultralytics / YOLO
- Plotly
- NumPy
- Pandas
- Pillow / OpenCV

---

## 5. Repository Structure

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
├── ui/
│   ├── app.py
│   ├── pipeline_services.py
│   ├── process_runner.py
│   ├── components.py
│   ├── ui_styles.py
│   └── assets/
│
├── docs/
│   ├── submissions/
│   ├── final/
│   ├── poster/
│   └── media/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Runtime folders such as `data/`, `checkpoints/`, `outputs/`, `runs/`, `logs/`, and `wandb/` are intentionally ignored by Git because they may contain large datasets, trained weights, generated embeddings, logs, or temporary runtime files.

---

## 6. Main Source Files

### `src/robotics_data_prep/filter_data.py`

Filters raw robotic trajectories and creates a filtered dataset CSV.

The filtering stage uses:

- robot-state / gripper-related signals
- end-effector or motion-related checks
- YOLO-based visual validation

Expected output:

```text
final_mamba_dataset.csv
```

Typical CSV status values:

```text
SUCCESS
FAILURE
SKIPPED
```

Important note: the current filtering focuses mainly on reliable pick/lift behavior and should not be described as complete real-world physical task validation for all possible manipulation phases.

---

### `src/openclip_embed.py`

Precomputes OpenCLIP embeddings for image frames in successful trajectories.

Instead of running OpenCLIP during every training epoch, the system computes visual features once and stores them as `.npy` files inside trajectory folders.

Expected embedding file:

```text
openclip_vitb32_laion2b_s34b_b79k_embeddings.npy
```

This design significantly reduces training time because the Mamba model can load precomputed visual features directly.

---

### `src/data_loader.py`

Loads the filtered robotic dataset and prepares training/evaluation samples.

Responsibilities:

- read `final_mamba_dataset.csv`
- filter rows by status, usually `SUCCESS`
- load precomputed OpenCLIP embeddings
- load robot trajectory files such as `obs_dict.pkl` and `policy_out.pkl`
- combine visual embeddings with robot state vectors
- construct fixed-length temporal windows
- pair each input window with the correct target action

Current default representation:

```text
visual_embedding_dim = 512
robot_state_dim      = 7
input_dim            = 519
action_dim           = 7
seq_length           = 10
```

---

### `src/mamba_model.py`

Defines the Mamba-based behavioral cloning model.

The model receives a sequence of multimodal vectors and predicts the next robotic action.

Main configurable parameters include:

```text
d_model
d_state
d_conv
expand
num_layers
dropout
action_dim
```

---

### `src/train.py`

Trains the Mamba behavioral cloning model.

The script supports:

- loading a filtered CSV
- loading precomputed OpenCLIP embeddings
- filtering by status, usually `SUCCESS`
- training with configurable hyperparameters
- early stopping
- checkpoint saving
- metric saving
- resume training from an interrupted checkpoint
- duplicate run protection using configuration matching

Typical output folder:

```text
checkpoints/<run_name>/
├── model.pth
├── last_checkpoint.pth
├── config.json
└── metrics.json
```

---

### `src/evaluate.py`

Evaluates a trained checkpoint.

The script loads a saved checkpoint and its configuration, rebuilds the Mamba model, evaluates predictions, computes metrics, and prints/saves prediction examples for the UI.

Common evaluation metrics:

```text
MSE
MAE
Per-dimension MAE
Evaluated sample count
Prediction examples
```

---

### `src/trajectory_scanner.py`

Scans dataset folders and detects structurally valid trajectory folders.

A valid trajectory folder is expected to include files such as:

```text
obs_dict.pkl
policy_out.pkl
image frames / image folders
```

This file validates folder structure. It does not determine whether a robotic task succeeded.

---

### `src/weights_registry.py`

Handles checkpoint metadata and helps detect previously trained configurations.

The registry is used to:

- record training runs
- compare run configurations
- avoid unnecessary retraining when a matching trained checkpoint already exists
- help connect the UI to saved model artifacts

---

## 7. Streamlit UI

The project includes a Streamlit user interface for running the full pipeline.

Start the UI:

```bash
streamlit run ui/app.py
```

The UI stages are:

```text
Dataset → YOLO → OpenCLIP → Model → Training → Evaluation → Results
```

### `ui/app.py`

Main UI controller.

Responsibilities:

- manage the active pipeline stage
- store UI state using `st.session_state`
- handle page navigation
- start long-running backend processes
- update stage status
- display Dataset, YOLO, OpenCLIP, Model, Training, Evaluation, and Results screens
- connect UI actions to backend commands

### `ui/pipeline_services.py`

Service layer between the UI and backend scripts.

Responsibilities:

- validate dataset folders
- detect filtered CSV files
- summarize CSV contents
- count OpenCLIP embeddings
- discover checkpoints
- build command-line calls for YOLO, OpenCLIP, training, and evaluation
- parse logs into progress values, metrics, and prediction examples

Important constants are defined here, for example:

```text
DEFAULT_DATASET_ROOT
DEFAULT_YOLO_WEIGHTS
DEFAULT_FILTER_CSV_NAME
EMBEDDING_FILENAME
CHECKPOINTS_DIR
```

### `ui/process_runner.py`

Runs long backend processes safely.

Responsibilities:

- launch subprocesses
- save process PID
- save log paths
- detect whether a process is still running
- send stop signals
- read log tails
- persist UI runtime state

Runtime state is saved under:

```text
outputs/ui_runtime/
```

UI logs are saved under:

```text
outputs/ui_runs/
```

### `ui/components.py`

Reusable UI components.

Includes:

- project brand header
- pipeline stepper
- stage headers
- metric cards
- progress bars
- log boxes
- source/config cards
- Plotly charts

### `ui/ui_styles.py`

CSS styling and layout rules for the Streamlit UI.

It controls the visual consistency of cards, buttons, graphs, logs, tables, and the Results screen.

---

## 8. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.10+
CUDA-enabled GPU
PyTorch with CUDA support
Conda environment recommended
```

The project was developed in a Conda environment named:

```text
mamba_proj
```

---

## 9. Required External Artifacts

Large artifacts are not stored in Git.

Before running the full pipeline, make sure the following exist locally:

### Dataset

A BridgeData-style raw trajectory folder, for example:

```text
/home/linuxu/Downloads/scripted_6_18/scripted_raw
```

### YOLO Weights

The UI expects the fixed YOLO weights file at:

```text
src/robotics_data_prep/best.pt
```

If this file is not committed to Git, place it manually at the expected path or update:

```text
DEFAULT_YOLO_WEIGHTS
```

inside:

```text
ui/pipeline_services.py
```

### Checkpoints

Training creates checkpoints under:

```text
checkpoints/<run_name>/
```

These folders are ignored by Git and should be stored locally or externally.

---

## 10. Step-by-Step CLI Usage

The Streamlit UI is the preferred way to run the project, but the pipeline can also be executed from the command line.

### Step 1: Filter trajectories

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

Only rows with:

```text
Status = SUCCESS
```

are used for the current training configuration.

---

### Step 2: Precompute OpenCLIP embeddings

```bash
python src/openclip_embed.py \
  --csv_path /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32
```

To recompute embeddings:

```bash
python src/openclip_embed.py \
  --csv_path /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32 \
  --overwrite
```

Expected embedding file inside successful trajectory folders:

```text
openclip_vitb32_laion2b_s34b_b79k_embeddings.npy
```

---

### Step 3: Train the Mamba model

Example training run:

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

Sanity-check training run:

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

---

### Step 4: Evaluate a trained checkpoint

Current UI evaluation command format:

```bash
python src/evaluate.py \
  --checkpoint_dir checkpoints/<run_name> \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --show_examples 5
```

Quick evaluation on a small number of batches, if supported by the local `evaluate.py`:

```bash
python src/evaluate.py \
  --checkpoint_dir checkpoints/<run_name> \
  --filtered_csv /home/linuxu/Downloads/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --max_batches 10 \
  --show_examples 5
```

---

## 11. Current Verified Configuration

The current working configuration used during development:

```json
{
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
  "train_split": 0.8,
  "random_seed": 42
}
```

---

## 12. Verified Development Results

A verified development run used:

```text
Successful trajectories: 4172
Training windows: 166718
Input shape: [10, 519]
Target shape: [7]
Embeddings loaded from .npy: 4172
Embeddings computed online: 0
Missing embedding trajectories: 0
```

Example longer run:

```text
Run name: mamba_k10_d128_lr0001_pick_success_v1
Best epoch: 9
Best validation loss: 0.004267

Evaluation:
Evaluated samples: 33344
MSE: 0.004267
MAE: 0.019006
Per-dimension MAE: [0.0062 0.0079 0.0111 0.0065 0.0072 0.034 0.0602]
```

These values document a verified development run and should be updated if a new final run is selected for the final submission.

---

## 13. Project Documentation

Suggested documentation folder structure:

```text
docs/
├── submissions/
│   ├── project_topic.pdf
│   ├── requirements.pdf
│   ├── sdd.pdf
│   └── stp.pdf
│
├── final/
│   ├── final_summary.md
│   └── std_results.md
│
├── poster/
│   ├── robomamba_poster.pdf
│   └── robomamba_poster.png
│
└── media/
    ├── screenshots/
    └── demo/
```

The final submission should include:

- all previous submission documents
- source code
- updated README
- final summary document
- STD system test results
- poster
- demo video link or demo video file, depending on file size

---

## 14. Poster and Media

The poster is stored in both PDF and PNG format:

```text
docs/poster/robomamba_poster.pdf
docs/poster/robomamba_poster.png
```

The PNG version is displayed directly in this README, while the PDF version is kept for full-quality viewing and submission.

Recommended screenshots folder, if UI screenshots are added later:

```text
docs/media/screenshots/
```

Recommended screenshots:

```text
01_dataset.png
02_yolo.png
03_openclip.png
04_model.png
05_training.png
06_evaluation.png
07_results.png
```

A demo video can be added later. If the video is small, it can be stored under `docs/media/demo/`. If it is large, it is better to upload it externally and place the link in `docs/final/final_summary.md`.

---

## 15. Important Git Notes

The following files and folders should not be committed:

```text
data/
datasets/
scripted_raw/
checkpoints/
outputs/
logs/
runs/
wandb/
*.pth
*.pt
*.ckpt
*.npy
*.npz
*.pkl
*.log
.streamlit/
```

This prevents large datasets, generated embeddings, model weights, logs, and local UI settings from being committed to Git.

If a generated file was already tracked by Git, remove it from Git tracking without deleting it locally:

```bash
git rm --cached <file_or_folder>
```

Example:

```bash
git rm -r --cached checkpoints outputs data
```

---

## 16. Known Limitations

- The system is an offline dataset-based imitation-learning pipeline.
- It does not control a physical robot in real time.
- It does not fine-tune OpenCLIP.
- It does not train YOLO from scratch.
- The YOLO filtering stage focuses mainly on successful pick/lift behavior.
- Full real-robot deployment is outside the current project scope.
- Advanced checkpoint integrity validation is limited; the UI checks available checkpoint files and metadata, but does not deeply validate every possible architecture mismatch.
- Current evaluation should be interpreted together with per-dimension errors and prediction examples, not only global MAE/MSE.

---

## 17. Suggested Future Work

Possible future improvements:

- trajectory-level train/validation split instead of random-window split
- additional sequence lengths such as `seq_length = 20`
- larger model dimension such as `d_model = 256`
- comparison with LSTM / Transformer baselines
- improved place-phase validation
- richer prediction-example saving with exact frame paths
- support for external artifact download instructions
- optional small sample dataset for reproducibility
- real robot deployment or closed-loop control experiments

---

## 18. Project Status

The project currently includes a working end-to-end pipeline:

```text
Raw trajectories
→ YOLO/gripper filtering
→ SUCCESS CSV
→ OpenCLIP precomputed embeddings
→ multimodal sequence construction
→ Mamba behavioral cloning training
→ checkpoint/config/metrics saving
→ evaluation
→ Streamlit UI results visualization
```

The system is intended as an academic final project prototype for multimodal robotic behavioral cloning using OpenCLIP and Mamba.
