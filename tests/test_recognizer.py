import pytest
from recon.recognizer import *


def test_base_recognizer(test_texts):
    recognizer = EntityRecognizer()

    with pytest.raises(NotImplementedError):
        recognizer.labels
    with pytest.raises(NotImplementedError):
        recognizer.predict(test_texts)


def test_spacy_recognizer(nlp, test_texts):
    ruler = nlp.create_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "SKILL", "pattern": "Machine learning"},
            {"label": "SKILL", "pattern": "researched"},
            {"label": "SKILL", "pattern": "AI"},
            {"label": "JOB_ROLE", "pattern": "Software Engineer"},
        ]
    )

    nlp.add_pipe(ruler)

    recognizer = SpacyEntityRecognizer(nlp)
    assert recognizer.labels == ["JOB_ROLE", "SKILL"]

    examples = list(recognizer.predict(test_texts))

    assert examples[0].text == test_texts[0]
    assert len(examples[0].spans) == 3
    assert len(examples[1].spans) == 2
