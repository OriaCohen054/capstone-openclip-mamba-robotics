import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

# Path to the base 'pnp' folder containing all trajectory groups
TRAJ_GROUP_PATH = "/home/linuxu/Documents/RoboMamba_OpenCLIP/data/pnp"

class MultiModalRoboticDataset(Dataset):
    """
    A PyTorch Dataset that loads multiple trajectories for generalization.
    It recursively searches for valid trajectory folders across all subdirectories.
    """
    def __init__(self, data_root, seq_length=10):
        self.seq_length = seq_length
        self.data_root = data_root
        
        print("Loading OpenCLIP model (ViT-B-32)...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.eval()
        
        self.samples = [] 
        self.traj_data = {} 
        
        # --- NEW RECURSIVE SEARCH LOGIC ---
        traj_folders = []
        # os.walk travels through all subdirectories inside data_root
        for root, dirs, files in os.walk(data_root):
            # A folder is a valid trajectory if it contains the pickle files and the images0 folder
            if "obs_dict.pkl" in files and "policy_out.pkl" in files and "images0" in dirs:
                traj_folders.append(root)
                
        traj_folders.sort()
        
        print(f"Recursive search found {len(traj_folders)} valid trajectory folders. Processing...")
        
        for traj_path in traj_folders:
            obs_path = os.path.join(traj_path, "obs_dict.pkl")
            policy_path = os.path.join(traj_path, "policy_out.pkl")
            images_dir = os.path.join(traj_path, "images0")
            
            # 1. Load States
            with open(obs_path, 'rb') as f:
                obs_data = pickle.load(f)
            states = np.vstack(obs_data['state']).astype(np.float32)
            
            # 2. Load Actions
            with open(policy_path, 'rb') as f:
                raw_actions = pickle.load(f)
            
            if isinstance(raw_actions[0], dict):
                action_key = next((k for k in ['action', 'actions', 'policy'] if k in raw_actions[0]), list(raw_actions[0].keys())[0])
                clean_actions = [step[action_key] for step in raw_actions]
                actions = np.vstack(clean_actions).astype(np.float32)
            else:
                actions = np.vstack(raw_actions).astype(np.float32)
                
            # 3. Load Image paths
            image_files = sorted([img for img in os.listdir(images_dir) if img.endswith('.jpg') or img.endswith('.png')])
            
            # Ensure synchronization
            num_steps = min(len(states), len(actions), len(image_files))
            
            # Only use trajectories longer than seq_length
            if num_steps > self.seq_length:
                self.traj_data[traj_path] = {
                    'states': states[:num_steps],
                    'actions': actions[:num_steps],
                    'image_files': image_files[:num_steps],
                    'images_dir': images_dir
                }
                
                for start_idx in range(num_steps - self.seq_length):
                    self.samples.append((traj_path, start_idx))
                    
        print("Dataset successfully initialized for Generalization.")
        print(f"Total valid trajectories processed: {len(self.traj_data)}")
        print(f"Total training windows extracted across all data: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, start_idx = self.samples[idx]
        traj = self.traj_data[traj_path]
        
        state_window = traj['states'][start_idx : start_idx + self.seq_length]
        state_window_tensor = torch.tensor(state_window, dtype=torch.float32)
        
        target_action = traj['actions'][start_idx + self.seq_length - 1]
        target_action_tensor = torch.tensor(target_action, dtype=torch.float32)
        
        window_image_files = traj['image_files'][start_idx : start_idx + self.seq_length]
        image_tensors = []
        
        for img_name in window_image_files:
            img_path = os.path.join(traj['images_dir'], img_name)
            image = Image.open(img_path).convert('RGB')
            img_tensor = self.clip_preprocess(image)
            
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            image_tensors.append(img_tensor)
            
        images_stacked = torch.stack(image_tensors)
        
        with torch.no_grad():
            visual_features = self.clip_model.encode_image(images_stacked)
            
        combined_features = torch.cat((visual_features, state_window_tensor), dim=1)
        
        return combined_features, target_action_tensor

def test_generalization_dataloader():
    print("Testing multi-trajectory initialization...")
    dataset = MultiModalRoboticDataset(TRAJ_GROUP_PATH, seq_length=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (fused_inputs, actions) in enumerate(dataloader):
        print("--- Generalization Batch Dimensions Verification ---")
        print(f"Fused Input Window Shape: {fused_inputs.shape}")
        print(f"Target Action Shape: {actions.shape}")
        break

if __name__ == "__main__":
    test_generalization_dataloader()