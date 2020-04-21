from pathlib import Path
from typing import Dict, List

import pytest
from recon.corpus import Corpus
from recon.loaders import read_jsonl
from recon.preprocess import SpacyPreProcessor
from recon.recognizer import SpacyEntityRecognizer
from recon.types import Example, TransformationCallbacks
from spacy.lang.en import English


@pytest.fixture()
def nlp():
    return English()


@pytest.fixture()
def spacy_preprocessor(nlp):
    return SpacyPreProcessor(nlp)


@pytest.fixture()
def test_texts():
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
    return Corpus.from_disk(base_path)


@pytest.fixture()
def example_corpus_processed() -> Corpus:
    """Fixture to load example train/dev/test data that has inconsistencies.
    
    Returns:
        Corpus: Example data
    """
    base_path = Path(__file__).parent.parent / "examples/data/skills"
    corpus = Corpus.from_disk(base_path)
    corpus.pipe_(
        [
            "recon.v1.fix_tokenization_and_spacing",
            "recon.v1.add_tokens",
            "recon.v1.upcase_labels",
            "recon.v1.filter_overlaps",
        ]
    )
    return corpus


@pytest.fixture()
def recognizer(nlp, example_corpus):
    patterns = []

    for e in example_corpus.all:
        for span in e.spans:
            patterns.append({"label": span.label, "pattern": span.text.lower()})

    nlp.add_pipe(nlp.create_pipe("entity_ruler", {"patterns": patterns}))

    recognizer = SpacyEntityRecognizer(nlp)
    return recognizer
