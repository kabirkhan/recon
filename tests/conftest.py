from pathlib import Path
from typing import Dict, List

import pytest
from recon.loaders import read_jsonl
from recon.stats import ner_stats
from recon.types import Example
from spacy.lang.en import English


@pytest.fixture()
def nlp():
    return English()


@pytest.fixture()
def test_texts():
    return [
        "Machine learning is the most researched area of AI.",
        "My title at work is Software Engineer even theough I mostly work on AI.",
    ]


@pytest.fixture()
def example_data() -> Dict[str, List[Example]]:
    """Fixture to load example train/dev/test data that has inconsistencies.
    
    ### Returns
    -----------
    (Dict[str, List[Example]]):
        Dataset containing the train/dev/test split
    """
    base_path = Path(__file__).parent.parent / "examples/data"
    return {
        "train": read_jsonl(base_path / "train.jsonl"),
        "dev": read_jsonl(base_path / "dev.jsonl"),
        "test": read_jsonl(base_path / "test.jsonl"),
    }
