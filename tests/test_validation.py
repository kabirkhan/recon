from typing import cast

import pytest
from recon.stats import get_ner_stats
from recon.types import Example, NERStats, Span, Token
from recon.validation import filter_overlaps, upcase_labels


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


def test_upcase_labels(example_corpus):
    stats = cast(NERStats, get_ner_stats(example_corpus.train))
    assert "skill" in stats.n_annotations_per_type
    assert "product" in stats.n_annotations_per_type
    assert "SKILL" in stats.n_annotations_per_type

    example_corpus._train.apply_("recon.v1.upcase_labels")
    fixed_stats = cast(NERStats, get_ner_stats(example_corpus.train))
    assert "skill" not in fixed_stats.n_annotations_per_type
    assert "product" not in fixed_stats.n_annotations_per_type


def test_filter_overlaps():
    def get_test_example(span_offsets):
        spans = []
        for so in span_offsets:
            spans.append(Span(text="x" * (so[1] - so[0]), start=so[0], end=so[1], label=so[2]))
        return Example(text="x" * 1500, spans=spans)

    def spans_to_offsets(spans):
        return [(span.start, span.end, span.label) for span in spans]

    test_entities = [(0, 5, "ENTITY"), (6, 10, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(0, 5, "ENTITY"), (6, 10, "ENTITY")]

    test_entities = [(0, 5, "ENTITY"), (5, 10, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(0, 5, "ENTITY"), (5, 10, "ENTITY")]

    test_entities = [(0, 5, "ENTITY"), (4, 10, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(4, 10, "ENTITY")]

    test_entities = [(0, 5, "ENTITY"), (0, 5, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(0, 5, "ENTITY")]

    test_entities = [(0, 5, "ENTITY"), (4, 11, "ENTITY"), (6, 20, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(0, 5, "ENTITY"), (6, 20, "ENTITY")]

    test_entities = [(0, 5, "ENTITY"), (4, 7, "ENTITY"), (10, 20, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(0, 5, "ENTITY"), (10, 20, "ENTITY")]

    test_entities = [(1368, 1374, "ENTITY"), (1368, 1376, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(1368, 1376, "ENTITY")]

    test_entities = [
        (12, 33, "ENTITY"),
        (769, 779, "ENTITY"),
        (769, 787, "ENTITY"),
        (806, 811, "ENTITY"),
    ]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [
        (12, 33, "ENTITY"),
        (769, 787, "ENTITY"),
        (806, 811, "ENTITY"),
    ]

    test_entities = [
        (189, 209, "ENTITY"),
        (317, 362, "ENTITY"),
        (345, 354, "ENTITY"),
        (364, 368, "ENTITY"),
    ]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [
        (189, 209, "ENTITY"),
        (317, 362, "ENTITY"),
        (364, 368, "ENTITY"),
    ]

    test_entities = [(445, 502, "ENTITY"), (461, 473, "ENTITY"), (474, 489, "ENTITY")]
    result = filter_overlaps(get_test_example(test_entities))
    assert spans_to_offsets(result.spans) == [(445, 502, "ENTITY")]
