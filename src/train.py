import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data_loader import MultiModalRoboticDataset, TRAJ_GROUP_PATH
from mamba_model import MultimodalMambaBC

# ---------------------------------------------------------
# Configuration & User Input Handling
# ---------------------------------------------------------
def get_training_config():
    print("--- Model Training Configuration ---")
    
    seq_length_input = input("Enter sequence length (K) [default: 10]: ")
    seq_length = int(seq_length_input) if seq_length_input.strip() else 10
    
    d_model_input = input("Enter d_model size [default: 128]: ")
    d_model = int(d_model_input) if d_model_input.strip() else 128
    
    patience_input = input("Enter Early Stopping patience [default: 5]: ")
    patience = int(patience_input) if patience_input.strip() else 5
    
    max_epochs_input = input("Enter max epochs [default: 100]: ")
    max_epochs = int(max_epochs_input) if max_epochs_input.strip() else 100
    
    batch_size_input = input("Enter batch size [default: 16]: ")
    batch_size = int(batch_size_input) if batch_size_input.strip() else 16
    
    return seq_length, d_model, patience, max_epochs, batch_size

# ---------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------
def main():
    seq_length, d_model, patience, max_epochs, batch_size = get_training_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nHardware setup complete. Using device: {device}")
    
    if device.type == "cpu":
        print("Warning: Training Mamba on CPU will be extremely slow.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_log_{timestamp}.txt"
    
    with open(log_filename, "w") as log_file:
        log_file.write(f"--- Training Run: {timestamp} ---\n")
        log_file.write(f"Dataset Path: {TRAJ_GROUP_PATH}\n")
        log_file.write(f"Hyperparameters: K={seq_length}, d_model={d_model}, patience={patience}, max_epochs={max_epochs}, batch={batch_size}\n\n")
        log_file.write("Epoch\tTrain Loss\tVal Loss\tStatus\n")
        log_file.write("-" * 55 + "\n")

        print("\nInitializing Dataset and extracting visual/kinematic features...")
        full_dataset = MultiModalRoboticDataset(TRAJ_GROUP_PATH, seq_length=seq_length)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"Data split complete: {train_size} train windows, {val_size} validation windows.")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print("Initializing MultimodalMambaBC architecture...")
        model = MultimodalMambaBC(input_dim=519, d_model=d_model, action_dim=7).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        counter = 0
        best_model_path = ""
        
        print("\nStarting real training loop...\n")
        
        for epoch in range(1, max_epochs + 1):
            
            # --- TRAIN PHASE ---
            model.train()
            total_train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- VALIDATION PHASE ---
            model.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            # --- EARLY STOPPING & CHECKPOINT CLEANUP ---
            status_msg = ""
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                
                # Delete old best model to save disk space
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # Save new best model
                best_model_path = f"mamba_k{seq_length}_dmodel{d_model}_loss{avg_val_loss:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                
                status_msg = f"New best model saved: {best_model_path}"
                print(f"--> {status_msg} (Old model deleted)")
            else:
                counter += 1
                status_msg = f"No improvement. Patience: {counter}/{patience}"
                print(f"--> {status_msg}")
                
                if counter >= patience:
                    stop_msg = f"Early stopping triggered at epoch {epoch}."
                    print(f"STOP: {stop_msg}")
                    log_file.write(f"\n{stop_msg}\n")
                    break
            
            log_file.write(f"{epoch}\t{avg_train_loss:.5f}\t\t{avg_val_loss:.5f}\t\t{status_msg}\n")
            
        summary_msg = f"\nTraining Complete.\nBest Validation Loss: {best_val_loss:.5f}\nBest Model Weight File saved as: {best_model_path}\n"
        print(summary_msg)
        log_file.write(summary_msg)

if __name__ == "__main__":
    main()