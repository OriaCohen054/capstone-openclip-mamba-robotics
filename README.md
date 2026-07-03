# RoboMamba: Multimodal Behavioral Cloning for a Robotic Arm Using Mamba and OpenCLIP

**Student:** Oria Cohen  
**Advisor:** Dr. Sharon Yalov-Handzel  
**Project:** Computer Science Final Project

---

## 1. Project Overview

This project implements an offline multimodal robotic imitation-learning pipeline for predicting the next robotic action from visual trajectory sequences and robot-state data.

The system combines:

- **YOLO-based trajectory filtering** for selecting usable robotic demonstrations and reducing visually invalid/silent-failure trajectories, using project-specific trained YOLO weights created from manually curated/annotated images.
- **OpenCLIP visual embeddings** for converting image frames into compact 512-dimensional visual feature vectors.
- **Robot state vectors** for adding non-visual robotic information.
- **Mamba sequence modeling** for learning temporal dependencies across consecutive timesteps.
- **Behavioral Cloning** for predicting the next continuous robotic action vector.
- **Streamlit UI** for operating the full pipeline end-to-end.

The implemented model receives a fixed-length sequence of multimodal features and predicts the next 7-dimensional robotic action.

> Important scope note: this project focuses on **offline next-action prediction from recorded robotic trajectories**. It does not deploy the policy on a physical robot and does not perform real-time closed-loop robot control.

---

## 2. Project Poster

The poster below summarizes the project motivation, solution, architecture, and main engineering challenges.

[View full poster as PDF](docs/poster/robomamba_poster.pdf)

![RoboMamba Project Poster](docs/poster/robomamba_poster.png)

---

## 3. Dataset

This project uses BridgeData-style robotic manipulation trajectories.

The dataset itself is **not included in this repository** because of its size.

Dataset source:

- BridgeData website: <https://rail-berkeley.github.io/bridgedata/>
- Bridge release data index: <https://rail.eecs.berkeley.edu/datasets/bridge_release/data/>

The project experiments were developed using the smaller scripted BridgeData release:

```text
scripted_6_18.zip  (~30GB)
```

After extracting the dataset, the UI expects a raw trajectory folder such as:

```text
/path/to/scripted_6_18/scripted_raw
```

The default local development path used in the UI code is:

```text
/home/linuxu/Downloads/scripted_6_18/scripted_raw
```

This path can be changed from the UI by choosing another dataset folder.

### Dataset Scope

The data comes from robotic manipulation trajectories. In this implementation, the pipeline focuses on:

- filtering usable successful trajectories,
- extracting OpenCLIP visual embeddings,
- synchronizing visual features with robot-state/action data,
- training a Mamba-based behavioral-cloning model for **next-action prediction**.

The project should not be described as a complete deployed pick-and-place robot controller. A safer description is:

> The system learns next-action prediction from recorded robotic manipulation trajectories using visual and robot-state context.

---

## 4. High-Level Pipeline

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
Mamba behavioral-cloning training
        ↓
Checkpoint + config + metrics
        ↓
Evaluation
        ↓
Streamlit Results visualization
```

Current input and target format used by the project:

```text
Input shape per sample:  [seq_length, 519]
519 = 512 OpenCLIP embedding + 7 robot-state features

Target shape per sample: [7]
Target = next robotic action vector
```

Example configurations used during development:

```text
seq_length = 10
d_model    = 128
batch_size = 16
learning_rate = 0.001
```

Latest documented run:

```text
seq_length = 12
d_model    = 512
batch_size = 64
learning_rate = 0.001
patience   = 7
```

---

## 5. Main Technologies

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

## 6. Repository Structure

The repository is organized around source code, UI code, and project documentation.

```text
capstone-openclip-mamba-robotics/
├── src/
│   ├── robotics_data_prep/
│   │   └── filter_data.py
│   ├── openclip_embed.py
│   ├── data_loader.py
│   ├── mamba_model.py
│   ├── train.py
│   ├── evaluate.py
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
│   ├── final/
│   │   ├── RoboMamba_Final_Summary.pdf
│   │   └── RoboMamba_STD_Final.pdf
│   │
│   ├── submissions/
│   │   ├── project_topic/
│   │   │   ├── original_hebrew_project_topic.pdf
│   │   │   └── RoboMamba_Project_Topic_Final.pdf
│   │   ├── requirements/
│   │   │   ├── original_hebrew_requirements_specification.pdf
│   │   │   └── RoboMamba_Requirements_Specification_Final.pdf
│   │   ├── literature_review/
│   │   ├── sdd/
│   │   │   ├── final_sdd.pdf
│   │   │   └── original_hebrew_sdd.pdf
│   │   ├── stp/
│   │   │   ├── RoboMamba_STP_Final.pdf
│   │   │   └── stp.pdf
│   │   └── presentation/
│   │
│   ├── poster/
│   │   ├── robomamba_poster.pdf
│   │   └── robomamba_poster.png
│   │
│   └── media/
│       └── screenshots/
│           └── demo/
│               ├── 01-ui-home-dataset-selection.png
│               ├── 02-ui-dataset-ready.png
│               ├── 03-ui-yolo-existing-csv.png
│               ├── 04-ui-yolo-completed-scan.png
│               ├── 05-ui-openclip-cache-ready.png
│               ├── 06-ui-model-existing-checkpoint.png
│               ├── 07-ui-model-new-training-configuration.png
│               ├── 08-ui-mamba-training-running.png
│               ├── 09-ui-mamba-training-completed.png
│               ├── 10-ui-evaluation-checkpoint-ready.png
│               ├── 11-ui-evaluation-completed-metrics.png
│               ├── 12-ui-results-metrics-and-charts.png
│               ├── 13-ui-results-prediction-example.png
│               └── 14-ui-results-evaluation-log.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Documentation Folder Notes

The `docs/` folder contains the final project documentation, previous submission material, poster files, and optional media.

The current documentation organization is:

```text
docs/final/          # Final summary and final STD
docs/submissions/    # Project topic, requirements, literature review, SDD, STP, presentation
docs/poster/         # Project poster PDF and PNG preview
docs/media/          # Demo screenshots and optional media assets
```

Some folders under `docs/submissions/` include both the original Hebrew submission and the final updated English version. This keeps the historical academic submissions available while also providing cleaner final English documentation.

Runtime folders such as `data/`, `checkpoints/`, `outputs/`, `runs/`, `logs/`, and `wandb/` are intentionally ignored by Git because they may contain large datasets, trained weights, generated embeddings, logs, or temporary runtime files.

## 7. Main Source Files

### `src/robotics_data_prep/filter_data.py`

Filters raw robotic trajectories and creates a filtered dataset CSV.

The filtering stage uses YOLO-based visual validation together with trajectory/state checks to identify usable trajectories.

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

Important note: the filtering stage helps curate reliable training data, but it should not be described as complete real-world task execution validation.

---

### `src/openclip_embed.py`

Precomputes OpenCLIP embeddings for image frames in successful trajectories.

Instead of running OpenCLIP during every training epoch, the system computes visual features once and stores them as `.npy` files inside trajectory folders.

Expected embedding file:

```text
openclip_vitb32_laion2b_s34b_b79k_embeddings.npy
```

This design reduces training time because the Mamba model can load precomputed visual features directly.

---

### `src/data_loader.py`

Loads the filtered robotic dataset and prepares training/evaluation samples.

Responsibilities:

- read `final_mamba_dataset.csv`,
- filter rows by status, usually `SUCCESS`,
- load precomputed OpenCLIP embeddings,
- load robot trajectory files such as `obs_dict.pkl` and `policy_out.pkl`,
- combine visual embeddings with robot state vectors,
- construct fixed-length temporal windows,
- pair each input window with the correct target action.

Main representation:

```text
visual_embedding_dim = 512
robot_state_dim      = 7
input_dim            = 519
action_dim           = 7
seq_length           = configurable, commonly 10 or 12
```

---

### `src/mamba_model.py`

Defines the Mamba-based behavioral-cloning model.

The model receives a sequence of multimodal vectors and predicts the next robotic action.

Main configurable parameters include:

```text
d_model
num_layers / Mamba depth-related settings
output_dim = 7
```

The exact architecture is defined in the source file.

---

### `src/train.py`

Runs Mamba training.

Responsibilities:

- parse training arguments,
- load the dataset through `data_loader.py`,
- create the Mamba model,
- train the behavioral-cloning model,
- compute training and validation losses,
- apply early stopping,
- save checkpoints,
- save configuration and training metadata.

Typical checkpoint output:

```text
checkpoints/<run_name>/
├── model.pth
├── config.json
├── metrics.json
└── last_checkpoint.pth
```

---

### `src/evaluate.py`

Runs evaluation for a selected trained checkpoint.

Responsibilities:

- load checkpoint configuration,
- rebuild the model architecture,
- load `model.pth`,
- run evaluation on the selected dataset,
- compute metrics such as MSE and MAE,
- print/save prediction examples for the UI Results screen.

---

### `src/weights_registry.py`

Provides checkpoint/weights registration or lookup utilities used by the training workflow.

The project also uses UI-side checkpoint discovery in `ui/pipeline_services.py` for displaying available checkpoint folders in the Streamlit interface.

---

## 8. Streamlit UI

The project includes a Streamlit interface for running the full pipeline without manually typing every command.

Run the UI with:

```bash
streamlit run ui/app.py
```

The UI is organized as:

```text
Dataset → YOLO → OpenCLIP → Model → Training → Evaluation → Results
```

### `ui/app.py`

Main Streamlit application.

Responsibilities:

- initialize and restore UI state,
- manage the current pipeline stage,
- render each page,
- handle user actions,
- launch backend processes,
- update stage statuses,
- display evaluation results.

---

### `ui/pipeline_services.py`

Service layer between the UI and backend scripts.

Responsibilities:

- validate dataset folders,
- detect `final_mamba_dataset.csv`,
- summarize CSV status values,
- count OpenCLIP embedding files,
- discover checkpoints,
- build YOLO/OpenCLIP/training/evaluation CLI commands,
- parse log files into progress values, metrics, and prediction examples.

---

### `ui/process_runner.py`

Runs long backend processes from the UI.

Responsibilities:

- start subprocesses,
- save PID and process metadata,
- write logs to `outputs/ui_runs/`,
- stop processes safely using Linux signals,
- restore current-process state after reopening the UI.

---

### `ui/components.py`

Reusable UI components.

Examples:

- project brand/header,
- stage stepper,
- metric cards,
- progress bars,
- log cards,
- Plotly charts,
- configuration blocks.

---

### `ui/ui_styles.py`

Central CSS/theme file for the Streamlit UI.

It keeps the UI visually consistent across all stages.

---

## 9. Installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.10+
Linux
CUDA-enabled NVIDIA GPU
PyTorch with CUDA support
```

The current requirements include CUDA-specific packages and Mamba/causal-conv wheels. If installation fails on another machine, the CUDA, PyTorch, and Python versions should be checked carefully.

---

## 10. Required External Artifacts

The repository does not include large datasets, generated embeddings, or trained checkpoints.

Before running the full pipeline, prepare the following locally:

### Dataset

Download and extract the BridgeData scripted dataset:

```text
scripted_6_18.zip (~30GB)
```

Expected selected folder:

```text
/path/to/scripted_6_18/scripted_raw
```

### YOLO Weights

The UI expects YOLO weights at:

```text
src/robotics_data_prep/best.pt
```

Because `.pt` files are ignored by Git, this file should be copied manually into the expected location when setting up the project.

### Optional Existing Checkpoints

If using an already trained model, place checkpoint folders under:

```text
checkpoints/<run_name>/
```

Each completed checkpoint folder should include at least:

```text
model.pth
config.json
metrics.json
```

---

## 11. Running the Pipeline from the UI

Start the UI:

```bash
streamlit run ui/app.py
```

Then follow the stages:

1. Select dataset folder.
2. Reuse or generate `final_mamba_dataset.csv` with YOLO filtering.
3. Reuse, resume, or recompute OpenCLIP embeddings.
4. Select an existing checkpoint or prepare a new training run.
5. Train or evaluate the selected model.
6. View metrics, charts, prediction examples, and logs in Results.

---

## 12. Running from Command Line

### YOLO Filtering

```bash
python src/robotics_data_prep/filter_data.py \
  --data_dir /path/to/scripted_6_18/scripted_raw \
  --weights src/robotics_data_prep/best.pt \
  --output final_mamba_dataset.csv
```

### OpenCLIP Embedding Generation

```bash
python src/openclip_embed.py \
  --csv_path /path/to/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --batch_size 32
```

### Training

```bash
python src/train.py \
  --filtered_csv /path/to/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --seq_length 12 \
  --d_model 512 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --max_epochs 50 \
  --patience 7 \
  --run_name mamba_k12_d512_lr0001_final
```

### Evaluation

```bash
python src/evaluate.py \
  --checkpoint_dir checkpoints/<run_name> \
  --filtered_csv /path/to/scripted_6_18/scripted_raw/final_mamba_dataset.csv \
  --status_filter SUCCESS \
  --show_examples 5
```

---

## 13. Evaluation Metrics

The evaluation stage reports quantitative and qualitative results.

Main metrics:

```text
MSE  - Mean Squared Error
MAE  - Mean Absolute Error
Per-dimension MAE
Number of evaluated samples
Saved prediction examples
```

The Results screen visualizes:

- overall MAE/MSE,
- per-dimension MAE,
- ground-truth action vs predicted action,
- prediction examples,
- action comparison table,
- evaluation log.

### Latest Documented Evaluation Run

One of the final documented evaluation runs used the following configuration:

```text
run_name      = mamba_k12_d512_lr0001_20260702_192016
seq_length    = 12
d_model       = 512
batch_size    = 64
learning_rate = 0.001
max_epochs    = 50
patience      = 7
```

The run completed successfully with early stopping and was evaluated using the selected trained checkpoint.

Evaluation summary:

```text
Valid trajectories loaded: 4172
Training windows created: 158374
Evaluated samples: 31675
Saved prediction examples: 50

MAE: 0.018485
MSE: 0.004496
```

Per-dimension MAE:

```text
Dim 1: 0.0061
Dim 2: 0.0096
Dim 3: 0.0107
Dim 4: 0.0068
Dim 5: 0.0070
Dim 6: 0.0318
Dim 7: 0.0573
```

The final action dimension had the highest error, which is consistent with the fact that gripper/open-close related behavior can be more discrete and harder to regress as a continuous value.

## 14. Generated Files Not Stored in Git

The following files and folders are generated during development or execution and are intentionally ignored:

```text
data/
datasets/
raw_data/
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
```

This keeps the repository lightweight and prevents large datasets, model weights, generated embeddings, and runtime logs from being committed.

---

## 15. Current Implementation Status

Implemented:

- Dataset folder validation.
- Existing filtered CSV detection and reuse.
- YOLO-based trajectory filtering integration.
- Manual YOLO data-curation workflow using representative trajectory images, Roboflow annotation, and trained YOLO weights.
- Conservative filtering behavior for selecting usable trajectories.
- YOLO stop, resume, restart, and reuse flows.
- OpenCLIP embedding generation, cache detection, reuse, resume, and recompute.
- Multimodal sequence loading with visual embeddings and robot-state vectors.
- Mamba-based behavioral cloning model.
- Training with configurable hyperparameters.
- Early stopping and checkpoint artifact saving.
- Checkpoint discovery and selection for evaluation.
- Evaluation with MAE/MSE and per-dimension MAE metrics.
- Prediction examples for qualitative review.
- Streamlit Results visualization.
- Process state persistence for long-running stages.
- Documentation of project topic, requirements, STP, STD, final summary, and poster.

Out of scope / not implemented as full system functionality:

- Real robot deployment.
- Real-time closed-loop robot control.
- Fine-tuning OpenCLIP.
- Full BridgeData benchmark evaluation across all tasks.
- Full comparison against RNN/LSTM/Transformer baselines.
- Standalone YOLO detector benchmarking as a separate research study.
- Advanced checkpoint compatibility validation across every possible model architecture change.

## 16. Project Documentation

The repository includes project documentation under `docs/`.

### Final Documentation

- [Final Project Summary](docs/final/RoboMamba_Final_Summary.pdf)
- [Software Test Design (STD)](docs/final/RoboMamba_STD_Final.pdf)

### Updated Submission Documents

- [Project Topic and Scope](docs/submissions/project_topic/RoboMamba_Project_Topic_Final.pdf)
- [Requirements Specification](docs/submissions/requirements/RoboMamba_Requirements_Specification_Final.pdf)
- [Software Test Plan (STP)](docs/submissions/stp/RoboMamba_STP_Final.pdf)

### Original Submission Material

Original Hebrew submission documents are kept under `docs/submissions/` for academic reference, including the original project topic, requirements, SDD, literature review, STP, and presentation material.

### Poster

- [Project Poster PDF](docs/poster/robomamba_poster.pdf)
- [Project Poster PNG Preview](docs/poster/robomamba_poster.png)

### Demo Video

A short demonstration video of the RoboMamba Streamlit pipeline is available here:

[Watch the RoboMamba Demo](https://youtu.be/O5coo0QvaRg)

### Demo Screenshots

A compact set of UI screenshots is stored under `docs/media/screenshots/demo/`.

- [UI home and dataset selection](docs/media/screenshots/demo/01-ui-home-dataset-selection.png)
- [Dataset ready state](docs/media/screenshots/demo/02-ui-dataset-ready.png)
- [YOLO existing CSV state](docs/media/screenshots/demo/03-ui-yolo-existing-csv.png)
- [YOLO completed scan](docs/media/screenshots/demo/04-ui-yolo-completed-scan.png)
- [OpenCLIP cache ready](docs/media/screenshots/demo/05-ui-openclip-cache-ready.png)
- [Model existing checkpoint](docs/media/screenshots/demo/06-ui-model-existing-checkpoint.png)
- [New training configuration](docs/media/screenshots/demo/07-ui-model-new-training-configuration.png)
- [Mamba training running](docs/media/screenshots/demo/08-ui-mamba-training-running.png)
- [Mamba training completed](docs/media/screenshots/demo/09-ui-mamba-training-completed.png)
- [Evaluation checkpoint ready](docs/media/screenshots/demo/10-ui-evaluation-checkpoint-ready.png)
- [Evaluation completed metrics](docs/media/screenshots/demo/11-ui-evaluation-completed-metrics.png)
- [Results metrics and charts](docs/media/screenshots/demo/12-ui-results-metrics-and-charts.png)
- [Results prediction example](docs/media/screenshots/demo/13-ui-results-prediction-example.png)
- [Results evaluation log](docs/media/screenshots/demo/14-ui-results-evaluation-log.png)

## 17. Notes for Reproducibility

To reproduce the pipeline, the user should provide:

1. The extracted BridgeData scripted dataset folder.
2. The YOLO `best.pt` weights file.
3. Python dependencies from `requirements.txt`.
4. CUDA-compatible PyTorch and Mamba installation.
5. Optional pretrained checkpoints, if evaluation should be run without retraining.

Large artifacts are intentionally excluded from Git and should be stored locally or externally.
