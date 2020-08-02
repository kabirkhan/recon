from typing import List

import catalogue
import spacy
from spacy.language import Language

loading_pipelines = catalogue.create("recon", "loading_pipelines", entry_points=True)
tokenizers = catalogue.create("recon", "tokenizers", entry_points=True)


@loading_pipelines.register("default")
def default_loading_pipeline() -> List[str]:
    return [
        "fix_tokenization_and_spacing",
        "add_tokens",
        "fix_annotations_format",
        "filter_overlaps",
        "json_to_examples",
    ]


@tokenizers.register("default")
def default_tokenizer() -> Language:
    return spacy.blank("en")
