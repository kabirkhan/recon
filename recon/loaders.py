"""Load a list of Example data from JSON represented records in 
the [Prodigy](https://prodi.gy) format.
"""

from pathlib import Path
from typing import List

import srsly

from .types import Example
from .validation import json_to_examples


def read_jsonl(path: Path) -> List[Example]:
    """Read annotations in JSONL file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    data = list(srsly.read_jsonl(path))
    return json_to_examples(data)


def read_json(path: Path) -> List[Example]:
    """Read annotations in JSON file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    data = srsly.read_json(path)
    return json_to_examples(data)
