"""
SPARSH: Subtype Prediction via Adaptive Recognition from Sparse Hematological Data

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import json


class SparseNN(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        n_classes: int = 18,
        dropout: float = 0.3,
        activation: str = "gelu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.dropout_rate = dropout
        self.activation_name = activation

        # Select activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")


        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Output layer (no activation — raw logits for loss functions)
        layers.append(nn.Linear(in_dim, n_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_config(self) -> Dict[str, Any]:

        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "n_classes": self.n_classes,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseNN":

        return cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            n_classes=config["n_classes"],
            dropout=config.get("dropout", 0.3),
            activation=config.get("activation", "gelu"),
        )

    def save(self, path: str, config: Optional[Dict] = None):
        """
        Save model weights and configuration.

        Args:
            path: Directory path to save model.
            config: Optional additional config to include.
        """
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), path / "model.pt")

        # Save config
        model_config = self.get_config()
        if config:
            model_config.update(config)

        with open(path / "config.json", "w") as f:
            json.dump(model_config, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseNN":
        """
        Load model from saved files.

        Args:
            path: Directory containing model.pt and config.json.
            device: Device to load model to.

        Returns:
            Loaded SparseNN model in eval mode.
        """
        from pathlib import Path

        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        model = cls.from_config(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)
        model.eval()

        return model




def apply_progressive_mask(
    x: torch.Tensor,
    mask_ratio: float,
    fill_value: float = 0.5,
) -> torch.Tensor:

    if mask_ratio <= 0:
        return x

    # Generate random mask
    mask = torch.rand_like(x) < mask_ratio

    # Apply mask
    x_masked = x.clone()
    x_masked[mask] = fill_value

    return x_masked


def get_mask_ratio(
    epoch: int,
    max_epochs: int,
    start: float = 0.97,
    end: float = 0.80,
) -> float:

    if max_epochs <= 1:
        return end

    progress = epoch / (max_epochs - 1)
    return start + (end - start) * progress
