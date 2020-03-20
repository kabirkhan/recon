from typing import Any, Dict, List
import catalogue
from .loaders import read_jsonl
from .pipelines import compose
from .types import Example
from .validation import (
    add_tokens,
    fix_annotations_format,
    filter_overlaps,
    json_to_examples,
)


class registry(object):
    loading_pipeline = catalogue.create("recon", "loading_pipeline", entry_points=True)


@registry.loading_pipeline.register("default")
def loading_pipeline(data: List[Dict[str, Any]]) -> List[Example]:
    return compose(add_tokens, fix_annotations_format, filter_overlaps, json_to_examples)
