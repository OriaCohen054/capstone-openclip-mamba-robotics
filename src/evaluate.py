import torch
import numpy as np
from torch.utils.data import DataLoader

# Import our custom modules
from data_loader import MultiModalRoboticDataset, TRAJ_GROUP_PATH
from mamba_model import MultimodalMambaBC

def evaluate_generalization():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # 2. Initialize Model and Load Weights
    print("\nLoading trained generalization weights...")
    model = MultimodalMambaBC(input_dim=519, d_model=128, action_dim=7).to(device)
    
    weights_path = "mamba_k10_dmodel128_loss0.0206.pth"
    try:
        # Added weights_only=True to resolve PyTorch security warning
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded weights from '{weights_path}'.")
    except FileNotFoundError:
        print(f"Error: Could not find '{weights_path}'. Make sure train.py ran successfully.")
        return

    # Set model to evaluation mode
    model.eval()

    # 3. Load Dataset
    print("\nLoading dataset for inference...")
    dataset = MultiModalRoboticDataset(TRAJ_GROUP_PATH, seq_length=10)
    
    # We use shuffle=True here so every time you run this script, 
    # you get to see different random windows from the data.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 4. Run Inference
    print("\n--- Inference Results (Random Windows) ---")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            # We will only test and print 3 random examples to avoid spamming the terminal
            if i >= 3:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Predict the action
            predicted_action = model(inputs)
            
            pred_np = predicted_action.cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]
            
            # Print side-by-side comparison
            np.set_printoptions(precision=4, suppress=True)
            print(f"\nExample {i+1}:")
            print(f"Ground Truth Action: {target_np}")
            print(f"Predicted Action:    {pred_np}")
            
            error = np.abs(target_np - pred_np)
            print(f"Mean Error (MAE):    {np.mean(error):.6f}")

if __name__ == "__main__":
    evaluate_generalization()