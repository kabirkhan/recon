from typing import Any, Dict, List
from pathlib import Path
import srsly
from timeit import default_timer as timer
from recon import Dataset, Example, Span, Stats, get_ner_stats


def test_ner_stats():

    base_path = (
        Path(__file__).parent.parent / "examples/data/skills"
    )
    data = srsly.read_jsonl(base_path / "train.jsonl")
    examples = to_rust(data)
    start = timer()
    stats = get_ner_stats(examples)
    end = timer()

    print(end - start, "seconds to run")

    assert isinstance(stats, Stats)
    assert stats.n_examples == 106 * 1_000
    assert stats.n_annotations == 243 * 1_000
    assert stats.n_examples_no_entities == 29 * 1_000
    assert stats.n_annotations_per_type["SKILL"] == 197 * 1_000
    assert stats.n_annotations_per_type["PRODUCT"] == 33 * 1_000
    assert stats.n_annotations_per_type["JOB_ROLE"] == 10 * 1_000
    assert stats.n_annotations_per_type["skill"] == 2 * 1_000
    assert stats.n_annotations_per_type["product"] == 1 * 1_000

    raise


def to_rust(data: Dict[str, Any]) -> List[Example]:
    examples = []

    for e in data:
        spans = []
        for s in e["spans"]:
            start, end = s["start"], s["end"]
            text = s.get("text", e["text"][start:end])
            spans.append(Span(text=text, start=start, end=end, label=s["label"]))
        examples.append(Example(text=e["text"], spans=spans))
    return examples * 1_000
