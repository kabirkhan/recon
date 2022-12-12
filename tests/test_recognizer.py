import pytest
from typing import List, cast

from spacy.language import Language
from spacy.pipeline import EntityRuler
from recon.recognizer import EntityRecognizer, SpacyEntityRecognizer


def test_base_recognizer(test_texts: List[str]):
    recognizer = EntityRecognizer()

    with pytest.raises(NotImplementedError):
        recognizer.labels
    with pytest.raises(NotImplementedError):
        recognizer.predict(test_texts)


def test_spacy_recognizer(nlp: Language, test_texts: List[str]):
    ruler = cast(EntityRuler, nlp.add_pipe("entity_ruler", name="entity_ruler"))
    ruler.add_patterns(
        [
            {"label": "SKILL", "pattern": "Machine learning"},
            {"label": "SKILL", "pattern": "researched"},
            {"label": "SKILL", "pattern": "AI"},
            {"label": "JOB_ROLE", "pattern": "Software Engineer"},
        ]
    )

    recognizer = SpacyEntityRecognizer(nlp)
    assert recognizer.labels == ["JOB_ROLE", "SKILL"]

    examples = list(recognizer.predict(test_texts))

    assert examples[0].text == test_texts[0]
    assert len(examples[0].spans) == 3
    assert len(examples[1].spans) == 2

    scores = recognizer.evaluate(examples, verbose=False)
    assert scores.ents_f == 1.0
