import torch
import torch.nn as nn

# Try to import Mamba. If it fails, it means the package is not installed in the conda env.
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm is not installed. Please rupython src/mamba_model.pyn: pip install mamba-ssm")
    Mamba = None

class MultimodalMambaBC(nn.Module):
    """
    Behavioral Cloning model using Mamba for sequence processing.
    Fuses visual embeddings (OpenCLIP) and proprioceptive state to predict actions.
    """
    def __init__(self, input_dim=519, d_model=128, action_dim=7):
        super().__init__()
        
        # 1. Input Projection: Map the 519-dimensional input to a smaller hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Mamba Block: The core State Space Model for sequence processing
        # We use a relatively small d_model (128) for the MVP to ensure fast training
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        
        # 3. Action Head: Project the final hidden state to the desired action dimension
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, x):
        # Expected input shape: (Batch, Sequence_Length, Input_Dim) -> e.g., (1, 10, 519)
        
        # Project input features
        x = self.input_proj(x)  # Shape becomes (B, 10, d_model)
        
        # Process sequence through Mamba
        x = self.mamba(x)       # Shape remains (B, 10, d_model)
        
        # Extract the feature representation at the LAST timestep 
        # because we want to predict the next immediate action at the end of the window
        last_step_features = x[:, -1, :]  # Shape becomes (B, d_model)
        
        # Predict the action vector
        predicted_action = self.action_head(last_step_features)  # Shape (B, 7)
        
        return predicted_action

def test_architecture():
    if Mamba is None:
        return
        
    print("Initializing Multimodal Mamba Architecture...")
    
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("Error: Mamba requires a CUDA-enabled GPU to run. The script cannot proceed on CPU.")
        return

    # Initialize the model and move it to the GPU
    model = MultimodalMambaBC(input_dim=519, d_model=128, action_dim=7).to(device)
    
    # Create a dummy tensor and move it to the exact same GPU
    dummy_input = torch.randn(1, 10, 519).to(device)
    
    print(f"Feeding dummy input of shape: {dummy_input.shape} on {dummy_input.device}")
    
    # Run the forward pass
    output = model(dummy_input)
    
    print(f"Model prediction shape: {output.shape}")
    print("Success: The network architecture is ready and matching the data dimensions.")

if __name__ == "__main__":
    test_architecture()