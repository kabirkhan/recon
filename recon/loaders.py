"""Load a list of Example data from JSON represented records in 
the [Prodigy](https://prodi.gy) format.
"""

from pathlib import Path
from typing import List

import srsly

from .util import registry
from .types import Example
from .validation import json_to_examples


def read_jsonl(path: Path, loading_pipeline: str = "default") -> List[Example]:
    """Read annotations in JSONL file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    data = list(srsly.read_jsonl(path))
    loading_pipeline = registry.loading_pipeline.get(loading_pipeline)
    return loading_pipeline(data)  # type: ignore


def read_json(path: Path, loading_pipeline: str = "default") -> List[Example]:
    """Read annotations in JSON file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    data = srsly.read_json(path)
    loading_pipeline = registry.loading_pipeline.get(loading_pipeline)
    return loading_pipeline(data)  # type: ignore
