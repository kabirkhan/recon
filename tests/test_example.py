from typing import Any, Dict
from recon.types import Example, Span



def test_example_init():
    dict_data: Dict[str, Any] = {"text": "text no spans", "spans": [{"start": 0, "end": 4, "label": "TEST"}]}
    example = Example(**dict_data)

    assert example.text == "text no spans"
    assert example.spans[0].start == 0
    assert example.spans[0].end == 4
    assert example.spans[0].label == "TEST"
    assert example.tokens is None


def test_example_init_text_only():
    dict_data: Dict[str, Any] = {"text": "text no spans"}
    example = Example(**dict_data)

    assert example.text == "text no spans"
    assert example.spans == []
    assert example.tokens is None
