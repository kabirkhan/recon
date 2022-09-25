from pathlib import Path
from typing import Union


def ensure_path(path: Union[str, Path]) -> Path:
    """Ensure string is converted to a Path.

    Args:
        path (Union[str, Path]): str or path. If string, it's converted to Path.
    Returns:
        Path: Pathlib Path object for the provided input
    """
    if isinstance(path, str):
        return Path(path)
    else:
        return path
