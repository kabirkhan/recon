from pathlib import Path
from typing import Any


def ensure_path(path: Any) -> Path:
    """Ensure string is converted to a Path.

    Args:
        path (Any): str or path. If string, it's converted to Path.
    Returns:
        Path: Pathlib Path object for the provided input
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("type of positional argument 'path' must be a string or Path")
    return Path(path) if isinstance(path, str) else path
