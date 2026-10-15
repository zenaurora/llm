"""Device selection with macOS MPS priority."""
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_available() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()