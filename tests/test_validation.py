import pytest
from recon.types import Example, Span, Token
from recon.util import registry
from recon.validation import *


@pytest.fixture()
def messy_data():
    return [
        {
            "text": "Denver, Colorado is a city.",
            "spans": [
                {"start": 0, "end": 6, "label": "GPE"},
                {"start": 0, "end": 16, "label": "LOC"},
            ],
            "meta": "Cities Data",
        }
    ]


def test_loading_pipeline(messy_data):
    loading_pipeline = registry.loading_pipelines.get("default")
    pipeline = loading_pipeline()
    fixed_examples = pipeline(messy_data)
    assert isinstance(fixed_examples[0], Example)
    assert isinstance(fixed_examples[0].spans[0], Span)
    assert isinstance(fixed_examples[0].tokens[0], Token)

    assert fixed_examples[0].spans[0].text == "Denver, Colorado"
    assert fixed_examples[0].spans[0].label == "LOC"


def test_fix_annotations_format(messy_data):
    fixed_data = fix_annotations_format(messy_data)

    assert fixed_data[0]["spans"][0]["text"] == "Denver"
    assert fixed_data[0]["meta"] == {"source": "Cities Data"}


# def test_remove_overlaps():
#     test_entities = [(0, 5, "ENTITY"), (6, 10, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(0, 5, "ENTITY"), (6, 10, "ENTITY")]

#     test_entities = [(0, 5, "ENTITY"), (5, 10, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(0, 5, "ENTITY"), (5, 10, "ENTITY")]

#     test_entities = [(0, 5, "ENTITY"), (4, 10, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(4, 10, "ENTITY")]

#     test_entities = [(0, 5, "ENTITY"), (0, 5, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(0, 5, "ENTITY")]

#     test_entities = [(0, 5, "ENTITY"), (4, 11, "ENTITY"), (6, 20, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(0, 5, "ENTITY"), (6, 20, "ENTITY")]

#     test_entities = [(0, 5, "ENTITY"), (4, 7, "ENTITY"), (10, 20, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(0, 5, "ENTITY"), (10, 20, "ENTITY")]

#     test_entities = [(1368, 1374, "ENTITY"), (1368, 1376, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(1368, 1376, "ENTITY")]

#     test_entities = [
#         (12, 33, "ENTITY"),
#         (769, 779, "ENTITY"),
#         (769, 787, "ENTITY"),
#         (806, 811, "ENTITY"),
#     ]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(12, 33, "ENTITY"), (769, 787, "ENTITY"), (806, 811, "ENTITY")]

#     test_entities = [
#         (189, 209, "ENTITY"),
#         (317, 362, "ENTITY"),
#         (345, 354, "ENTITY"),
#         (364, 368, "ENTITY"),
#     ]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(189, 209, "ENTITY"), (317, 362, "ENTITY"), (364, 368, "ENTITY")]

#     test_entities = [(445, 502, "ENTITY"), (461, 473, "ENTITY"), (474, 489, "ENTITY")]
#     result = remove_overlapping_entities(test_entities)
#     assert result == [(445, 502, "ENTITY")]
