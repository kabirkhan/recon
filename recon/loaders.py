"""Load a list of Example data from JSON represented records in 
the [Prodigy](https://prodi.gy) format.
"""

import functools
from pathlib import Path
from typing import Any, Dict, List, Union

import srsly

from .types import Example, OperationState, OperationStatus, Transformation
from .operations import registry
from .registry import loading_pipelines


def read_jsonl(
    path: Path, loading_pipeline: Union[str, List[str]] = "default"
) -> List[Example]:
    """Read annotations in JSONL file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    if isinstance(loading_pipeline, str):
        loading_pipeline = loading_pipelines.get(loading_pipeline)()

    data = list(srsly.read_jsonl(path))
    data = fix_annotations_format(data)
    examples = json_to_examples(data)
    return examples


def read_json(
    path: Path, loading_pipeline: str = "default"
) -> List[Example]:
    """Read annotations in JSON file format
    
    Args:
        path (Path): Path to data
    
    Returns:
        List[Example]: List of Examples
    """
    if isinstance(loading_pipeline, str):
        loading_pipeline = loading_pipelines.get(loading_pipeline)()

    data = list(srsly.read_jsonl(path))
    data = fix_annotations_format(data)
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


def fix_annotations_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix annotations format for a consistent dataset
    
    Args:
        data (List[Dict[str, Any]]): List of JSON Examples
    
    Returns:
        List[Dict[str, Any]]: List of JSON Examples with corrected formatting
    """
    for e in data:
        if "meta" not in e:
            e["meta"] = {}
        if isinstance(e["meta"], list) or isinstance(e["meta"], str):
            e["meta"] = {"source": e["meta"]}

        for s in e["spans"]:
            if "text" not in s:
                s["text"] = e["text"][s["start"] : s["end"]]
            s["label"] = s["label"].upper()
    return data