# my_library/config.py
import torch

def _detect_device() -> torch.device:
    """Automatically detects the best available hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon support
        return torch.device("mps")
    else:
        return torch.device("cpu")

_GLOBAL_DEVICE = _detect_device()

def set_device(device: str | torch.device):
    global _GLOBAL_DEVICE
    _GLOBAL_DEVICE = torch.device(device) if isinstance(device, str) else device

def get_device() -> torch.device:
    return _GLOBAL_DEVICE