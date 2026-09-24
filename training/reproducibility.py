"""
SPARSH-next reproducibility utilities.

set_deterministic_mode seeds Python, NumPy and PyTorch and requests
deterministic kernels. PyTorch is run with warn_only=True, so an operation
without a deterministic implementation warns instead of failing: runs on the
same hardware and library versions should agree closely, but bitwise identity
is not guaranteed. PYTHONHASHSEED only takes effect if it is set before Python
starts, so it is exported in the PBS job scripts rather than here.
"""

import hashlib
import logging
import os
import platform
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic_mode(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None and torch.cuda.is_available():
        logger.warning("CUBLAS_WORKSPACE_CONFIG is not set; GPU matrix products may not be deterministic "
                       "(the PBS scripts set it to :4096:8)")
    logger.info(f"Seeds set to {seed}")


def generate_run_id(seed: int = 42, timestamp: Optional[str] = None) -> str:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{hashlib.md5(f'{seed}_{timestamp}'.encode()).hexdigest()[:8]}"


def get_environment_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        meta["cuda"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0)
    for name in ("sklearn", "pandas", "matplotlib"):
        try:
            meta[name] = __import__(name).__version__
        except ImportError:
            pass
    return meta
