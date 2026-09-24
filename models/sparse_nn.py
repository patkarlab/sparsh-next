"""
SPARSH-next model.

A feed-forward classifier (Linear -> BatchNorm -> GELU -> Dropout blocks and a
linear head) whose first step is an explicit INPUT ENCODING of the beta values.
Inputs may contain NaN: a CpG with no data (masked in training, failed on the
array, or without ONT reads).

Input encodings
---------------
midpoint : missing -> 0.5, as in SPARSH v0.1.0 (default). The first-layer signal
           grows with the number of observed CpGs. That is harmless, and gave the
           best-calibrated probabilities in testing, as long as training covers
           the whole coverage range seen at inference (coverage_mode "random").
           With v0.1.0's narrow 3-20% schedule, dense arrays or denser ONT
           samples fall outside the scale the network was trained on.
scaled   : centre observed values at 0.5 (x - 0.5), set missing CpGs to 0 and
           divide by the sample's observed fraction (the inverted-dropout rule),
           so the expected first-layer input is the same at every coverage.
           Offered as an experiment: similar accuracy in synthetic tests,
           slightly worse calibration at 20-30% coverage.

Known limitation (both encodings): an observed value of exactly 0.5, such as one
of two reads methylated, is encoded like a missing CpG. Under the read model this
concerns about 1-3% of observed CpGs at 10-30% coverage.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

ENCODINGS = ("midpoint", "scaled")


class SparseNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (1024, 512, 256),
        n_classes: int = 2,
        dropout: float = 0.45,
        activation: str = "gelu",
        input_encoding: str = "midpoint",
        min_observed_fraction: float = 1e-3,
    ):
        super().__init__()
        if input_encoding not in ENCODINGS:
            raise ValueError(f"input_encoding must be one of {ENCODINGS}")
        activations = {"gelu": nn.GELU, "relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1)}
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.n_classes = int(n_classes)
        self.dropout_rate = float(dropout)
        self.activation_name = activation
        self.input_encoding = input_encoding
        self.min_observed_fraction = float(min_observed_fraction)

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), activations[activation](), nn.Dropout(dropout)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.n_classes))
        self.network = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Turn beta values with NaN for missing CpGs into network input."""
        if self.input_encoding == "midpoint":
            return torch.nan_to_num(x, nan=0.5)
        observed = ~torch.isnan(x)
        fraction = observed.float().mean(dim=1, keepdim=True).clamp_min(self.min_observed_fraction)
        centred = torch.where(observed, x - 0.5, torch.zeros_like(x))
        return centred / fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.encode(x))

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "n_classes": self.n_classes,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "input_encoding": self.input_encoding,
            "min_observed_fraction": self.min_observed_fraction,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseNN":
        return cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            n_classes=config["n_classes"],
            dropout=config.get("dropout", 0.45),
            activation=config.get("activation", "gelu"),
            input_encoding=config.get("input_encoding", "midpoint"),
            min_observed_fraction=config.get("min_observed_fraction", 1e-3),
        )


def load_model(weights_path, model_config: Dict[str, Any], device: str = "cpu") -> SparseNN:
    """Build a SparseNN from its config and load weights; returned in eval mode."""
    model = SparseNN.from_config(model_config)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model: SparseNN, X: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Logits for a float32 matrix that may contain NaN. The model must be in eval mode."""
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.n_classes), dtype=np.float32)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
