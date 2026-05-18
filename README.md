# Capstone: OpenCLIP + Mamba for Robotic Imitation Learning

This project implements a pipeline for learning visual robotic sequences using OpenCLIP embeddings and a Mamba-based sequence model.

## Project Pipeline

1. Filter raw robotic trajectories using gripper-state signals and YOLO-based vision verification.
2. Extract OpenCLIP embeddings from trajectory image frames.
3. Build temporal windows from embeddings and robot-state data.
4. Train a Mamba sequence model for behavioral cloning / next-action prediction.
5. Evaluate the trained model using validation metrics.

## Repository Structure

docs/  
Project documents, reports, requirements, SDD, and presentation materials.

src/  
Main source code for data preparation, embedding extraction, model training, and evaluation.

src/robotics_data_prep/  
Scripts for filtering and preparing raw BridgeData trajectories.

## Main Scripts

- src/robotics_data_prep/filter_data.py  
  Filters raw trajectories and creates a CSV dataset.

- src/openclip_embed.py  
  Extracts OpenCLIP visual embeddings from image frames.

- src/data_loader.py  
  Loads processed trajectory data and builds temporal training samples.

- src/mamba_model.py  
  Defines the Mamba-based sequence model.

- src/train.py  
  Trains the model.

- src/evaluate.py  
  Evaluates a trained checkpoint.

## Ignored Files

Large datasets, model checkpoints, experiment outputs, and temporary files are not tracked in Git.
