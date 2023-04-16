from pathlib import Path
from typing import Dict, List, cast

import pytest
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import EntityRuler

from recon.corpus import Corpus
from recon.corrections import *  # noqa: F401, F403
from recon.loaders import read_jsonl
from recon.preprocess import SpacyPreProcessor
from recon.recognizer import SpacyEntityRecognizer
from recon.tokenization import *  # noqa: F401, F403
from recon.types import Example
from recon.validation import *  # noqa: F401, F403


@pytest.fixture()
def nlp() -> English:
    return English()


@pytest.fixture()
def spacy_preprocessor(nlp: Language) -> SpacyPreProcessor:
    return SpacyPreProcessor(nlp=nlp)


@pytest.fixture()
def test_texts() -> List[str]:
    return [
        "Machine learning is the most researched area of AI.",
        "My title at work is Software Engineer even theough I mostly work on AI.",
    ]


@pytest.fixture()
def example_data() -> Dict[str, List[Example]]:
    """Fixture to load example train/dev/test data that has inconsistencies.

    Returns:
        Dict[str, List[Example]]: Dataset containing the train/dev/test split
    """
    base_path = Path(__file__).parent.parent / "examples/data/skills"
    return {
        "train": read_jsonl(base_path / "train.jsonl"),
        "dev": read_jsonl(base_path / "dev.jsonl"),
        "test": read_jsonl(base_path / "test.jsonl"),
    }


@pytest.fixture()
def example_corpus() -> Corpus:
    """Fixture to load example train/dev/test data that has inconsistencies.

    Returns:
        Corpus: Example data
    """
    base_path = Path(__file__).parent.parent / "examples/data/skills"
    return Corpus.from_disk(base_path, name="test_corpus")


@pytest.fixture()
def example_corpus_processed() -> Corpus:
    """Fixture to load example train/dev/test data that has inconsistencies.

    Returns:
        Corpus: Example data
    """
    base_path = Path(__file__).parent.parent / "examples/data/skills"
    corpus = Corpus.from_disk(base_path, name="test_corpus")
    corpus.pipe_(
        [
            "recon.fix_tokenization_and_spacing.v1",
            "recon.add_tokens.v1",
            "recon.upcase_labels.v1",
            "recon.filter_overlaps.v1",
        ]
    )
    return corpus


@pytest.fixture()
def recognizer(nlp: Language, example_corpus: Corpus) -> SpacyEntityRecognizer:
    patterns = []

    for e in example_corpus.all:
        for span in e.spans:
            patterns.append({"label": span.label, "pattern": span.text.lower()})

    ruler = cast(EntityRuler, nlp.add_pipe("entity_ruler", name="entity_ruler"))
    ruler.add_patterns(patterns)
    recognizer = SpacyEntityRecognizer(nlp)
    return recognizer
