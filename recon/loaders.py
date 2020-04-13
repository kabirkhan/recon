"""Load a list of Example data from JSON represented records in 
the [Prodigy](https://prodi.gy) format.
"""

import functools
from pathlib import Path
from typing import Any, Dict, List, Union

import srsly

from .operations import registry
from .registry import loading_pipelines
from .types import Example, OperationState, OperationStatus, Transformation


def read_jsonl(path: Path) -> List[Example]:
    """Read annotations in JSONL file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of examples
    """
    data = srsly.read_jsonl(path)
    examples = json_to_examples(data)
    return examples


def read_json(path: Path) -> List[Example]:
    """Read annotations in JSON file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of examples
    """
    data = srsly.read_jsonl(path)
    examples = json_to_examples(data)
    return examples


def json_to_examples(data: List[Dict[str, Any]]) -> List[Example]:
    """Convert List of Dicts to List of typed Examples
    
    Args:
        data (List[Dict[str, Any]]): Input List of Dicts to convert
    
    Returns:
        List[Example]: List of typed Examples
    """
    return [Example(**example) for example in data]
