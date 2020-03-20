from typing import Any, Callable, Dict, List

import catalogue
import spacy
from spacy.language import Language

from .pipelines import compose
from .types import Example
from .validation import (
    add_tokens,
    filter_overlaps,
    fix_annotations_format,
    json_to_examples,
)


class registry(object):
    loading_pipelines = catalogue.create(
        "recon", "loading_pipelines", entry_points=True
    )
    tokenizers = catalogue.create("recon", "tokenizers", entry_points=True)


@registry.tokenizers.register("default")
def tokenizer() -> Language:
    return spacy.blank("en")


@registry.loading_pipelines.register("default")
def loading_pipeline(tokenizer: str = "default") -> Callable:
    def add_tokens_wrapper(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nlp = registry.tokenizers.get(tokenizer)()
        return add_tokens(nlp, data)

    return compose(
        add_tokens_wrapper, fix_annotations_format, filter_overlaps, json_to_examples
    )
