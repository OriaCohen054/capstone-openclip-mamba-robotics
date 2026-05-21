import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class MultimodalMambaBC(nn.Module):
    """
    Behavioral Cloning model based on Mamba sequence modeling.

    The model receives a temporal sequence of fused features:
    OpenCLIP visual embedding + robot state features.

    Input shape:
        (batch_size, seq_length, input_dim)

    Output shape:
        (batch_size, action_dim)
    """

    def __init__(
        self,
        input_dim: int = 519,
        d_model: int = 128,
        action_dim: int = 7,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize the multimodal Mamba model.

        Args:
            input_dim: Dimension of the fused input vector at each timestep.
            d_model: Hidden dimension used by the Mamba blocks.
            action_dim: Output action dimension.
            d_state: Internal SSM state size used by Mamba.
            d_conv: Local convolution width used by Mamba.
            expand: Expansion factor used inside the Mamba block.
            num_layers: Number of stacked Mamba blocks.
            dropout: Dropout probability applied after projections and blocks.
        """
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Install it inside the correct "
                "environment before training or evaluating the model."
            )

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        # Project fused visual + robot-state features into the Mamba hidden space.
        self.input_proj = nn.Linear(input_dim, d_model)

        # Dropout is optional. When dropout=0.0, this layer has no effect.
        self.dropout = nn.Dropout(dropout)

        # Stack one or more Mamba blocks.
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # LayerNorm improves training stability when stacking sequence blocks.
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # Predict the action vector from the final timestep representation.
        self.action_head = nn.Linear(d_model, action_dim)

        # Store model configuration for debugging and checkpoint metadata.
        self.model_config = {
            "input_dim": input_dim,
            "d_model": d_model,
            "action_dim": action_dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        Args:
            x: Input tensor with shape (batch_size, seq_length, input_dim).

        Returns:
            Predicted action tensor with shape (batch_size, action_dim).
        """
        if x.dim() != 3:
            raise ValueError(
                "Expected input tensor with shape "
                "(batch_size, seq_length, input_dim)."
            )

        # Map the fused input features into the model hidden dimension.
        x = self.input_proj(x)
        x = self.dropout(x)

        # Process the sequence through stacked Mamba blocks.
        for mamba_layer, norm_layer in zip(self.mamba_layers, self.norm_layers):
            residual = x
            x = mamba_layer(x)
            x = self.dropout(x)

            # Residual connection helps preserve information across layers.
            x = norm_layer(x + residual)

        # Use the final timestep to predict the action at the end of the window.
        last_step_features = x[:, -1, :]

        # Produce the final continuous action prediction.
        predicted_action = self.action_head(last_step_features)

        return predicted_action


def test_architecture() -> None:
    """
    Run a small architecture sanity check with dummy input.
    """
    if Mamba is None:
        print("mamba_ssm is not installed. Cannot run the architecture test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        print("Warning: Mamba is expected to run on a CUDA-enabled GPU.")

    model = MultimodalMambaBC(
        input_dim=519,
        d_model=128,
        action_dim=7,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=1,
        dropout=0.0,
    ).to(device)

    dummy_input = torch.randn(2, 10, 519).to(device)
    output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")

    expected_shape = (2, 7)

    if output.shape == expected_shape:
        print("Success: Model output shape is correct.")
    else:
        print(f"Error: Expected output shape {expected_shape}, got {output.shape}.")


if __name__ == "__main__":
    test_architecture()